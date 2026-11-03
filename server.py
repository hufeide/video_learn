#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频教程生成器后端

流程 (两阶段):
    阶段一: 输入视频网址
            → yt-dlp 下载音频 (MP3)
            → ffmpeg 按 120 秒切分
            → 本地 MOSS-Transcribe-Diarize (localhost:9005) 逐段转录
            → 保存 transcript.txt, 暂停并展示给用户确认
    阶段二: 用户点击「继续生成教程」
            → 调用 spark-x2.6 LLM (按提示词文件规则) 生成学习教程 Markdown

缓存策略:
    - 网址-任务对应表 (url_map.json): 按视频标识规范化匹配
      (B站 BV 号 / av 号 / YouTube ID, 忽略跟踪参数差异)
    - 同一视频 → 同一 task_id → 同一任务目录, 不重复下载 mp3
    - 若该任务下 transcript 已存在, 阶段一直接跳过下载/切分/转录, 暂停等待确认
    - 若 tutorial 已存在且使用的提示词相同, 阶段二直接返回已有教程
    - /api/regenerate 用于阶段二: 只生成教程, 不重新转录
    - /api/status 在服务重启后可从磁盘恢复任务状态

产物命名 (位于 work/{task_id}/ 下):
    - {mp3基名}.mp3            音频 (保留)
    - {mp3基名}.transcript.txt 转录文本
    - {mp3基名}.tutorial.md    学习教程
    - url.txt                  原始网页地址

导出:
    - GET /api/files/{task_id}         列出可导出文件
    - GET /api/download/{task_id}/{name} 下载单个文件
    - GET /api/export/{task_id}        一键打包 zip 导出

接口:
    GET  /                     前端页面
    GET  /api/prompts          可选提示词文件列表
    GET  /api/tasks            网址-任务对应表
    POST /api/generate         {"url": "...", "prompt": "..."}  阶段一: 转录
    POST /api/regenerate       {"task_id": "...", "prompt": "..."}  阶段二: 生成教程
    GET  /api/status/{task_id} 轮询进度与结果 (支持磁盘恢复)

用法:
    python3 server.py [--host 0.0.0.0] [--port 8000]
"""

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
from pathlib import Path

import requests
import yt_dlp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "work"
DEFAULT_PROMPT = "video_prompt_0.md"
URL_MAP_FILE = BASE_DIR / "url_map.json"
URL_MAP_LOCK = threading.Lock()

# 从 .env 读取配置（环境变量优先于 .env）
load_dotenv(BASE_DIR / ".env")

# ---- 本地语音转录服务 (vllm, MOSS-Transcribe-Diarize) ----
TRANSCRIPT_API = os.environ.get("TRANSCRIPT_API", "http://localhost:9005/v1/audio/transcriptions")
TRANSCRIPT_MODEL = os.environ.get("TRANSCRIPT_MODEL", "/home/administator/workspace/hub/MOSS-Transcribe-Diarize")
SEGMENT_TIME = int(os.environ.get("SEGMENT_TIME", "120"))  # 秒

# ---- LLM (OpenAI 兼容接口, spark-x2.6) ----
LLM_BASE = os.environ.get("LLM_BASE", "https://maas-api.cn-huabei-1.xf-yun.com/v1")
LLM_KEY = os.environ.get("LLM_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "spark-x2.6")

# ---- 转录分块总结的每块字符数 (中文 ~1字≈1token, 留余量) ----
CHUNK_CHARS = 8000

app = FastAPI(title="视频教程生成器")

# task_id -> {status, stage, progress, message, title, markdown, transcript, error}
TASKS = {}
TASKS_LOCK = threading.Lock()


# ---------------- 工具函数 ----------------

def update_task(task_id, **kw):
    with TASKS_LOCK:
        if task_id in TASKS:
            TASKS[task_id].update(kw)


def make_task_id(url):
    """由视频网址生成唯一 task_id (md5 前 12 位)"""
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:12]


def normalize_url(url):
    """提取视频唯一标识, 去除跟踪参数差异 (同一视频不同参数 → 同一 key)"""
    # B站 BV 号
    m = re.search(r'(BV[0-9A-Za-z]+)', url)
    if m:
        return 'bilibili:' + m.group(1)
    # B站 av 号
    m = re.search(r'av(\d+)', url)
    if m:
        return 'bilibili:av' + m.group(1)
    # YouTube youtu.be / shorts
    m = re.search(r'youtu\.be/([\w-]+)', url)
    if m:
        return 'youtube:' + m.group(1)
    m = re.search(r'youtube\.com/(?:watch\?(?:v=)?|shorts/|embed/|live/)([\w-]+)', url)
    if m:
        return 'youtube:' + m.group(1)
    # 其他站点: 用原始 URL
    return url


def load_url_map():
    """读取网址-任务映射表"""
    if URL_MAP_FILE.exists():
        try:
            return json.loads(URL_MAP_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_url_map(data):
    URL_MAP_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def migrate_legacy_tasks():
    """
    服务启动时调用: 把旧任务目录 (md5(url) 命名) 迁移进网址-任务对应表
    同一视频的多个旧任务目录会被合并, 选中最优的一个 (有 tutorial 优先, 最新优先)
    """
    if load_url_map():
        return
    groups = {}  # key -> [(work_dir, url), ...]
    if WORK_DIR.exists():
        for work in sorted(WORK_DIR.iterdir()):
            if not work.is_dir():
                continue
            url = ""
            tp = work / "url.txt"
            if tp.exists():
                url = tp.read_text(encoding="utf-8").strip()
            if url:
                key = normalize_url(url)
            else:
                # 无 url.txt 的旧任务: 尝试从 mp3 文件名提取视频标识
                key = None
                for mp3 in work.glob("*.mp3"):
                    m = re.search(r'(BV[0-9A-Za-z]+)', mp3.name)
                    if m:
                        key = 'bilibili:' + m.group(1)
                        break
                if key is None:
                    key = work.name  # 无法识别, 保留为独立条目
            groups.setdefault(key, []).append((work, url))
    if not groups:
        return
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    data = {}
    for key, items in groups.items():
        # 排序: 有 tutorial 优先, 其次修改时间最新
        def rank(item):
            work, _ = item
            has_tu = bool(list(work.glob("*.tutorial.md"))) or (work / "tutorial.md").exists()
            return (has_tu, work.stat().st_mtime)
        items.sort(key=rank, reverse=True)
        winner, _ = items[0]
        urls = list(dict.fromkeys(u for _, u in items if u))
        data[key] = {
            "key": key,
            "task_id": winner.name,
            "urls": urls,
            "created": now,
            "last_used": now,
            "migrated": True,
        }
    save_url_map(data)
    print(f"已迁移 {len(data)} 个旧任务到 url_map.json")


def get_or_create_task_id(url):
    """
    由视频网址获取任务 ID (网址-任务对应表):
    - 按规范化后的视频标识匹配, 同一视频不同跟踪参数视为同一任务
    - 已有映射时直接返回映射表中的 task_id (复用任务目录)
    - 首次访问则用 md5(key) 生成 task_id 并创建映射记录
    """
    key = normalize_url(url)
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    with URL_MAP_LOCK:
        data = load_url_map()
        entry = data.get(key)
        if entry:
            # 已有任务: 复用其 task_id
            task_id = entry.get("task_id") or hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
            urls = entry.get("urls", [])
            if url not in urls:
                urls.append(url)
                entry["urls"] = urls
            entry["last_used"] = now
            save_url_map(data)
            return task_id
        # 新视频: 生成 task_id 并创建映射
        task_id = hashlib.md5(key.encode("utf-8")).hexdigest()[:12]
        data[key] = {
            "key": key,
            "task_id": task_id,
            "urls": [url],
            "created": now,
            "last_used": now,
        }
        save_url_map(data)
        return task_id


def list_prompt_files():
    """列出 BASE_DIR 下所有可用的 .md 提示词文件"""
    return sorted(p.name for p in BASE_DIR.glob("*.md"))


def resolve_prompt(name):
    """解析提示词文件名 → 绝对路径; 非法或不存在则回退到默认"""
    name = (name or "").strip() or DEFAULT_PROMPT
    p = (BASE_DIR / name).resolve()
    # 防止路径穿越
    if not str(p).startswith(str(BASE_DIR)) or p.suffix != ".md" or not p.exists():
        return BASE_DIR / DEFAULT_PROMPT
    return p


def find_transcript(work):
    """查找已有转录文件 (兼容新旧命名)"""
    for p in sorted(work.glob("*.transcript.txt")):
        return p
    old = work / "transcript.txt"
    return old if old.exists() else None


def find_tutorial(work):
    """查找已有教程文件 (兼容新旧命名)"""
    for p in sorted(work.glob("*.tutorial.md")):
        return p
    old = work / "tutorial.md"
    return old if old.exists() else None


def find_mp3(work):
    """查找任务目录下的 mp3 文件"""
    for p in sorted(work.glob("*.mp3")):
        return p
    return None


def load_title(work):
    tp = work / "title.txt"
    return tp.read_text(encoding="utf-8").strip() if tp.exists() else ""


def llm_chat(messages, temperature=0.3, max_tokens=8192, timeout=180):
    """调用 OpenAI 兼容 chat/completions, 返回 assistant content"""
    url = f"{LLM_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(url, headers=headers, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def download_audio(url, out_dir, progress_cb):
    """yt-dlp 下载并转码为 MP3, 返回 (mp3_path, title)"""
    opts = {
        "ignoreerrors": False,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
        "windowsfilenames": True,
        "format": "bestaudio/best",
        "outtmpl": os.path.join(out_dir, "%(title)s [%(id)s].%(ext)s"),
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "0"},
            {"key": "FFmpegMetadata"},
        ],
        "progress": True,
        "noprogress": False,
        "newline": True,
        "logger": _ProgressLogger(progress_cb),
    }
    # 代理: YouTube 等境外站点需要 (通过 .env 的 YTDLP_PROXY 配置)
    proxy = os.environ.get("YTDLP_PROXY", "").strip()
    if proxy:
        opts["proxy"] = proxy
    # cookies: YouTube 常需要登录态 (PO token / 人机验证)
    #   方式1: 从浏览器读取 (需浏览器在运行), 如 YTDLP_COOKIES_BROWSER=chrome
    #   方式2: 指定 cookies.txt 文件, 如 YTDLP_COOKIE_FILE=/path/cookies.txt
    cookies_browser = os.environ.get("YTDLP_COOKIES_BROWSER", "").strip()
    cookie_file = os.environ.get("YTDLP_COOKIE_FILE", "").strip()
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    elif cookie_file and os.path.exists(cookie_file):
        opts["cookiefile"] = cookie_file
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
    title = info.get("title", "video")
    mp3_path = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
    if not os.path.exists(mp3_path):
        mp3s = [os.path.join(out_dir, f) for f in os.listdir(out_dir) if f.lower().endswith(".mp3")]
        if mp3s:
            mp3_path = max(mp3s, key=os.path.getmtime)
    return mp3_path, title


class _ProgressLogger:
    """把 yt-dlp 进度百分比映射到 0~40 区间"""
    def __init__(self, cb):
        self.cb = cb
        self.last = 0

    def _emit(self, pct):
        val = int(pct * 0.4)
        if val > self.last:
            self.last = val
            self.cb(val, f"下载中 {pct:.0f}%")

    def debug(self, msg):
        if msg.startswith("[download]") and "%" in msg:
            try:
                self._emit(float(msg.split("%")[0].split(" ")[-1]))
            except (ValueError, IndexError):
                pass

    def info(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        pass


def split_audio(mp3_path, parts_dir, progress_cb):
    """ffmpeg 按 SEGMENT_TIME 秒切分 (-c copy), 返回分段文件列表"""
    parts_dir.mkdir(parents=True, exist_ok=True)
    pattern = os.path.join(parts_dir, "audio_part_%03d.mp3")
    cmd = ["ffmpeg", "-y", "-i", mp3_path, "-f", "segment",
           "-segment_time", str(SEGMENT_TIME), "-c", "copy", pattern]
    proc = subprocess.run(cmd, capture_output=True)  # bytes 模式, 避免中文文件名导致解码错误
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffmpeg 切分失败: {stderr[-500:]}")
    parts = sorted(parts_dir.glob("audio_part_*.mp3"))
    progress_cb(50, f"切分完成，共 {len(parts)} 段")
    return parts


def transcribe_part(part_path):
    """调用本地转录服务, 返回 text"""
    with open(part_path, "rb") as f:
        r = requests.post(
            TRANSCRIPT_API,
            data={"model": TRANSCRIPT_MODEL, "response_format": "json", "temperature": "0"},
            files={"file": (part_path.name, f, "audio/mpeg")},
            timeout=300,
        )
    r.raise_for_status()
    return r.json()["text"]


def transcribe_all(parts, progress_cb):
    """逐段转录并拼接, 返回完整转录文本"""
    texts = []
    total = len(parts)
    for i, part in enumerate(parts, 1):
        try:
            texts.append(transcribe_part(part))
        except Exception as e:
            raise RuntimeError(f"第 {i}/{total} 段转录失败: {e}")
        pct = 50 + int((i / total) * 35)
        progress_cb(pct, f"转录中 {i}/{total} 段")
    return "\n".join(texts)


def summarize_transcript(transcript, progress_cb, prompt_file=None):
    """分块提取要点, 再合并生成最终学习教程 Markdown"""
    prompt_file = prompt_file or (BASE_DIR / DEFAULT_PROMPT)
    prompt_rules = prompt_file.read_text(encoding="utf-8") if prompt_file.exists() else ""

    # 1) 分块提取要点
    chunks = []
    for i in range(0, len(transcript), CHUNK_CHARS):
        chunks.append(transcript[i:i + CHUNK_CHARS])

    points = []
    for ci, chunk in enumerate(chunks, 1):
        progress_cb(85 + int(ci / max(len(chunks), 1) * 10), f"要点提取 {ci}/{len(chunks)} 块")
        msgs = [
            {"role": "system", "content": (
                "你是一名视频内容提炼助手。请从下面的视频转录文本片段中提取关键信息要点。"
                "只输出要点列表, 每条一行, 以「- 」开头, 每条不超过60字, 保留关键概念/数据/结论, 去除口头禅与重复内容。不要输出其他内容。"
            )},
            {"role": "user", "content": f"转录文本片段 {ci}/{len(chunks)}:\n\n{chunk}"},
        ]
        points.append(llm_chat(msgs, temperature=0.2, timeout=180))

    points_text = "\n\n".join(f"【第{ci}部分要点】\n{p}" for ci, p in enumerate(points, 1))

    # 2) 合并生成最终教程
    msgs = [
        {"role": "system", "content": (
            prompt_rules +
            "\n\n你现在根据以上各部分要点, 生成一份完整的视频学习教程 Markdown。"
            "要求: 严格按照上面的输出格式与风格要求; 内容忠于转录要点, 不要编造; "
            "结构清晰, 重点加粗; 输出纯正的 Markdown 文本, 不要输出 HTML 或代码块包裹。"
        )},
        {"role": "user", "content": f"视频完整转录文本共 {len(transcript)} 字, 各部分要点如下:\n\n{points_text}"},
    ]
    progress_cb(97, "正在生成教程…")
    return llm_chat(msgs, temperature=0.3, max_tokens=8192, timeout=300)


# ---------------- 任务流水线 ----------------

def run_pipeline(task_id, url, prompt_name, mode="transcribe"):
    """
    两阶段流水线:

    mode="transcribe": 下载 → 切分 → 转录 → 保存 transcript
        完成后暂停, 状态置为 transcript_ready, 等待用户确认后继续
    mode="tutorial":   读取已有 transcript → 生成教程 (复用已有教程时直接返回)
    """
    work = WORK_DIR / task_id
    prompt_file = resolve_prompt(prompt_name)
    try:
        os.makedirs(work, exist_ok=True)

        # 保存原始网页地址
        if url:
            (work / "url.txt").write_text(url, encoding="utf-8")

        if mode == "tutorial":
            # ---- 阶段二: 生成教程 (必须已有 transcript) ----
            transcript_path = find_transcript(work)
            if not transcript_path:
                raise RuntimeError("该任务还没有转录文本，无法生成教程")
            transcript = transcript_path.read_text(encoding="utf-8")
            update_task(task_id, transcript=transcript)
            mp3 = find_mp3(work)
            mp3_basename = mp3.stem if mp3 else transcript_path.stem.replace(".transcript", "")
            update_task(task_id, title=load_title(work))
        else:
            # ---- 阶段一: 下载 → 切分 → 转录 → 保存 → 暂停等待确认 ----
            transcript_path = find_transcript(work)
            if transcript_path and transcript_path.stat().st_size > 0:
                # 复用已有转录, 直接进入待确认状态
                transcript = transcript_path.read_text(encoding="utf-8")
                mp3 = find_mp3(work)
                mp3_basename = mp3.stem if mp3 else transcript_path.stem.replace(".transcript", "")
                update_task(task_id, title=load_title(work))
            else:
                # 下载音频
                update_task(task_id, status="downloading", stage="下载音频", progress=2, message="开始下载…")
                mp3_path, title = download_audio(url, str(work),
                                                 lambda p, m: update_task(task_id, progress=p, message=m))
                mp3_basename = Path(mp3_path).stem
                update_task(task_id, title=title)
                (work / "title.txt").write_text(title, encoding="utf-8")

                # 切分音频
                update_task(task_id, status="splitting", stage="切分音频", progress=45, message="ffmpeg 切分中…")
                parts = split_audio(mp3_path, work / "parts",
                                    lambda p, m: update_task(task_id, progress=p, message=m))

                # 语音转录
                update_task(task_id, status="transcribing", stage="语音转录", progress=52,
                            message=f"共 {len(parts)} 段，开始转录…")
                transcript = transcribe_all(parts, lambda p, m: update_task(task_id, progress=p, message=m))

                # 保存转录 (以 mp3 基名为前缀)
                transcript_path = work / f"{mp3_basename}.transcript.txt"
                transcript_path.write_text(transcript, encoding="utf-8")

            update_task(task_id, transcript=transcript)

            # 转录完成, 暂停等待用户确认
            update_task(task_id, status="transcript_ready", stage="转录完成", progress=100,
                        message="转录完成，确认无误后点击「继续生成教程」")
            return

        # ---- 教程文件路径 ----
        if mp3_basename:
            tutorial_path = work / f"{mp3_basename}.tutorial.md"
        else:
            tutorial_path = find_tutorial(work) or work / "tutorial.md"

        # ---- 教程是否可复用 (存在且提示词相同) ----
        prompt_used_path = work / "prompt_used.txt"
        prompt_used = prompt_used_path.read_text(encoding="utf-8").strip() if prompt_used_path.exists() else ""
        if tutorial_path.exists() and prompt_used == prompt_name:
            markdown = tutorial_path.read_text(encoding="utf-8")
            update_task(task_id, markdown=markdown, status="done", stage="完成", progress=100,
                        message="复用已有教程 ✅")
            return

        # ---- 生成教程 ----
        update_task(task_id, status="summarizing", stage="生成教程", progress=85,
                    message="调用 LLM 生成学习教程…")
        markdown = summarize_transcript(transcript,
                                        lambda p, m: update_task(task_id, progress=p, message=m),
                                        prompt_file)

        tutorial_path.write_text(markdown, encoding="utf-8")
        prompt_used_path.write_text(prompt_name, encoding="utf-8")
        update_task(task_id, markdown=markdown, status="done", stage="完成", progress=100,
                    message="教程生成完毕 ✅")
    except Exception as e:
        update_task(task_id, status="error", stage="出错", progress=0, message=str(e),
                    error=str(e))
    finally:
        # 清理切分中间产物; 保留 mp3 / transcript / tutorial
        shutil.rmtree(work / "parts", ignore_errors=True)


# ---------------- 导出 ----------------

def get_export_files(task_id):
    """列出任务下可导出的文件 (mp3 / transcript / tutorial)"""
    work = WORK_DIR / task_id
    files = []
    if not work.exists():
        return files
    for p in sorted(work.iterdir()):
        if (p.is_file() and p.suffix in (".mp3", ".txt", ".md")
                and p.name not in ("prompt_used.txt", "title.txt")):
            files.append({"name": p.name, "size": p.stat().st_size})
    return files


# ---------------- API ----------------

class GenerateRequest(BaseModel):
    url: str
    prompt: str = ""


class RegenerateRequest(BaseModel):
    task_id: str
    prompt: str = ""


@app.get("/api/prompts")
def api_prompts():
    """返回可选的提示词文件列表及默认项"""
    return {"prompts": list_prompt_files(), "default": DEFAULT_PROMPT}


@app.get("/prompt_files/{name}")
def api_prompt_file(name: str):
    """返回指定提示词文件的原文 (供前端预览)"""
    p = resolve_prompt(name)
    if not p.exists() or p.suffix != ".md":
        raise HTTPException(404, "提示词文件不存在")
    return FileResponse(p, media_type="text/markdown; charset=utf-8")


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "请输入 http/https 开头的视频链接")
    prompt_name = (req.prompt or "").strip() or DEFAULT_PROMPT
    resolve_prompt(prompt_name)

    task_id = get_or_create_task_id(url)  # 网址-任务对应表 (同一视频 → 同一任务)
    with TASKS_LOCK:
        existing = TASKS.get(task_id)
        if existing and existing["status"] not in ("error",):
            return {"task_id": task_id, "reused": "task_exists"}  # 已有任务, 前端直接轮询
        TASKS[task_id] = {"status": "queued", "stage": "排队中", "progress": 0,
                          "message": "任务已创建", "url": url}
    threading.Thread(target=run_pipeline, args=(task_id, url, prompt_name, "transcribe"),
                     daemon=True).start()
    return {"task_id": task_id}


@app.post("/api/regenerate")
def api_regenerate(req: RegenerateRequest):
    """继续/重新生成教程: 读取已有 transcript, 只调用 LLM 生成教程"""
    task_id = req.task_id.strip()
    if not task_id:
        raise HTTPException(400, "缺少 task_id")
    prompt_name = (req.prompt or "").strip() or DEFAULT_PROMPT
    resolve_prompt(prompt_name)

    if not find_transcript(WORK_DIR / task_id):
        raise HTTPException(400, "该任务还没有转录文本，无法生成教程")
    with TASKS_LOCK:
        if task_id not in TASKS:
            TASKS[task_id] = {"status": "queued", "stage": "排队中", "progress": 0,
                              "message": "任务已创建"}
    threading.Thread(target=run_pipeline, args=(task_id, "", prompt_name, "tutorial"),
                     daemon=True).start()
    return {"task_id": task_id}


@app.get("/api/files/{task_id}")
def api_files(task_id: str):
    """列出任务下可导出的文件"""
    return {"files": get_export_files(task_id)}


@app.get("/api/download/{task_id}/{filename}")
def api_download(task_id: str, filename: str):
    """下载任务下的单个文件"""
    work = (WORK_DIR / task_id).resolve()
    p = (work / filename).resolve()
    if not str(p).startswith(str(work)) or not p.is_file():
        raise HTTPException(404, "文件不存在")
    return FileResponse(p, filename=p.name)


@app.get("/api/export/{task_id}")
def api_export(task_id: str):
    """一键打包导出全部文件 (zip)"""
    files = get_export_files(task_id)
    if not files:
        raise HTTPException(404, "任务下没有可导出的文件")
    work = WORK_DIR / task_id
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(work / f["name"], arcname=f["name"])
    buf.seek(0)
    return Response(
        buf.getvalue(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{task_id}.zip"'},
    )


@app.get("/api/tasks")
def api_tasks():
    """返回网址-任务对应表 (含文件状态)"""
    entries = []
    with TASKS_LOCK:
        tasks = dict(TASKS)
    for key, info in load_url_map().items():
        task_id = info.get("task_id", "")
        work = WORK_DIR / task_id
        entry = dict(info)
        entry["files"] = [f["name"] for f in get_export_files(task_id)]
        task = tasks.get(task_id)
        if task:
            entry["status"] = task["status"]
            entry["title"] = task.get("title", "")
        else:
            # 服务重启后内存任务丢失, 从磁盘推断
            tp = find_transcript(work)
            tu = find_tutorial(work)
            if tu:
                entry["status"] = "done"
            elif tp:
                entry["status"] = "transcript_ready"
            else:
                entry["status"] = "idle"
            entry["title"] = load_title(work)
        entries.append(entry)
    return {"tasks": entries}


@app.get("/api/status/{task_id}")
def api_status(task_id: str):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
    if not task:
        # 内存中没有 (如服务重启), 从磁盘恢复任务状态
        work = WORK_DIR / task_id
        tp = find_transcript(work)
        tu = find_tutorial(work)
        if not tp:
            raise HTTPException(404, "任务不存在")
        task = {
            "status": "transcript_ready",
            "stage": "转录完成",
            "progress": 100,
            "message": "转录完成，确认无误后点击「继续生成教程」",
            "title": load_title(work),
            "transcript": tp.read_text(encoding="utf-8"),
        }
        if tu:
            task["status"] = "done"
            task["stage"] = "完成"
            task["message"] = "教程生成完毕 ✅"
            task["markdown"] = tu.read_text(encoding="utf-8")
        with TASKS_LOCK:
            TASKS[task_id] = task
    return task


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "index.html")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    migrate_legacy_tasks()
    print(f"服务启动: http://{args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
