#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频教程生成器后端

流程: 输入视频网址
      → yt-dlp 下载音频 (MP3)
      → ffmpeg 按 120 秒切分
      → 本地 MOSS-Transcribe-Diarize (localhost:9005) 逐段转录
      → 拼接完整转录文本
      → 调用 spark-x2.6 LLM (按 vedio_prompt.md 规则) 生成学习教程 Markdown

用法:
    python3 server.py [--host 0.0.0.0] [--port 8000]

接口:
    GET  /                     前端页面
    POST /api/generate         {"url": "..."}  -> {"task_id": "..."}
    GET  /api/status/{task_id} 轮询进度与结果
"""

import argparse
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests
import yt_dlp
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).resolve().parent
WORK_DIR = BASE_DIR / "work"
PROMPT_FILE = BASE_DIR / "vedio_prompt.md"

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


def split_audio(mp3_path, parts_dir, progress_cb):
    """ffmpeg 按 SEGMENT_TIME 秒切分 (-c copy), 返回分段文件列表"""
    parts_dir.mkdir(parents=True, exist_ok=True)
    pattern = os.path.join(parts_dir, "audio_part_%03d.mp3")
    cmd = ["ffmpeg", "-y", "-i", mp3_path, "-f", "segment",
           "-segment_time", str(SEGMENT_TIME), "-c", "copy", pattern]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 切分失败: {proc.stderr[-500:]}")
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


def summarize_transcript(transcript, progress_cb):
    """分块提取要点, 再合并生成最终学习教程 Markdown"""
    prompt_rules = PROMPT_FILE.read_text(encoding="utf-8") if PROMPT_FILE.exists() else ""

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

def run_pipeline(task_id, url):
    work = WORK_DIR / task_id
    try:
        update_task(task_id, status="downloading", stage="下载音频", progress=2, message="开始下载…")
        os.makedirs(work, exist_ok=True)

        mp3_path, title = download_audio(url, str(work),
                                         lambda p, m: update_task(task_id, progress=p, message=m))
        update_task(task_id, title=title)

        update_task(task_id, status="splitting", stage="切分音频", progress=45, message="ffmpeg 切分中…")
        parts = split_audio(mp3_path, work / "parts",
                            lambda p, m: update_task(task_id, progress=p, message=m))

        update_task(task_id, status="transcribing", stage="语音转录", progress=52,
                    message=f"共 {len(parts)} 段，开始转录…")
        transcript = transcribe_all(parts, lambda p, m: update_task(task_id, progress=p, message=m))

        (work / "transcript.txt").write_text(transcript, encoding="utf-8")
        update_task(task_id, transcript=transcript)

        update_task(task_id, status="summarizing", stage="生成教程", progress=85, message="调用 LLM 生成学习教程…")
        markdown = summarize_transcript(transcript, lambda p, m: update_task(task_id, progress=p, message=m))

        (work / "tutorial.md").write_text(markdown, encoding="utf-8")
        update_task(task_id, markdown=markdown, status="done", stage="完成", progress=100,
                    message="教程生成完毕 ✅")
    except Exception as e:
        update_task(task_id, status="error", stage="出错", progress=0, message=str(e),
                    error=str(e))
    finally:
        # 清理中间音频文件, 保留 transcript/tutorial
        shutil.rmtree(work / "parts", ignore_errors=True)
        for f in work.glob("*.mp3"):
            try:
                f.unlink()
            except OSError:
                pass


# ---------------- API ----------------

class GenerateRequest(BaseModel):
    url: str


@app.post("/api/generate")
def api_generate(req: GenerateRequest):
    url = req.url.strip()
    if not url.startswith(("http://", "https://")):
        raise HTTPException(400, "请输入 http/https 开头的视频链接")
    task_id = uuid.uuid4().hex[:12]
    with TASKS_LOCK:
        TASKS[task_id] = {"status": "queued", "stage": "排队中", "progress": 0, "message": "任务已创建"}
    threading.Thread(target=run_pipeline, args=(task_id, url), daemon=True).start()
    return {"task_id": task_id}


@app.get("/api/status/{task_id}")
def api_status(task_id: str):
    with TASKS_LOCK:
        task = TASKS.get(task_id)
    if not task:
        raise HTTPException(404, "任务不存在")
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
    print(f"服务启动: http://{args.host}:{args.port}")
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
