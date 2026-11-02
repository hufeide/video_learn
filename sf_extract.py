#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoundFerry 的 Python 调用脚本（等价于 App 内「粘贴链接 → 导出 MP3」）。

用法：
    python sf_extract.py <视频链接> [输出目录]
    python sf_extract.py https://www.bilibili.com/video/BVxxx
    python sf_extract.py https://youtu.be/xxx ./mp3

依赖：
    pip install yt-dlp        （版本与 App 内置 yt-dlp 一致即可）
    系统需安装 ffmpeg（转码 MP3 用；App 打包里带的 ffmpeg.exe 也可以）

说明：
    - 支持 B站 / YouTube 及 yt-dlp 支持的 1800+ 站点（和 App 的 yt-dlp 引擎一致）
    - 抖音(douyin)在 App 里走的是自研 Chrome CDP 引擎，yt-dlp 对其支持不稳定，
      遇到风控时可能失败，这是平台限制而非脚本问题。
"""

import sys
import os
import re
import json
import shutil

try:
    import yt_dlp
except ImportError:
    sys.exit("缺少依赖，请先执行: pip install yt-dlp")


# ---- 可配置项（与 App 默认设置保持一致）----
AUDIO_QUALITY = "0"          # App 音质设置: "0"=VBR V0最佳, "320"/"192"/"128"
# ffmpeg 路径：优先环境变量，其次 PATH 中的 ffmpeg（App 打包里带的 ffmpeg.exe 也可指向）
FFMPEG_LOCATION = os.environ.get("FFMPEG_LOCATION") or shutil.which("ffmpeg") or "ffmpeg"
DEFAULT_OUTPUT_DIR = os.environ.get("SF_OUTPUT_DIR", os.path.join(os.getcwd(), "mp3"))


def extract_mp3(url, output_dir=None, quality=AUDIO_QUALITY):
    """
    输入视频链接，下载并转码为 MP3，返回 mp3 文件路径。

    :param url: 视频链接（B站/YouTube/通用站点）
    :param output_dir: 输出目录，默认 DEFAULT_OUTPUT_DIR
    :param quality: "0" / "320" / "192" / "128"
    :return: dict { "path": mp3绝对路径, "title": 标题, "uploader": 作者, "duration": 时长, "size": 字节 }
    """
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # 与 App 的 ytdlp.js 参数一一对应
    opts = {
        "ignoreerrors": False,
        "noplaylist": True,               # 只处理单条，不展开合集
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
        "windowsfilenames": True,
        "trim_filenames": 120,
        "ffmpeg_location": FFMPEG_LOCATION,
        "format": "bestaudio/best",      # -x 取音频
        "outtmpl": os.path.join(output_dir, "%(title)s [%(id)s].%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",   # -x --audio-format mp3
                "preferredcodec": "mp3",
                "preferredquality": quality,   # --audio-quality
            },
            {"key": "FFmpegMetadata"},         # 写入 ID3 元数据（--embed-metadata）
        ],
        "progress": True,
        "noprogress": False,
        "newline": True,
    }

    # 可选：从 cookies.txt 读取登录态（需要登录的内容才用）
    cookie_file = os.environ.get("SF_COOKIE_FILE")
    if cookie_file and os.path.exists(cookie_file):
        opts["cookiefile"] = cookie_file

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)

    # 产物路径：postprocessor 转码后的实际文件
    mp3_path = ydl.prepare_filename(info).rsplit(".", 1)[0] + ".mp3"
    if not os.path.exists(mp3_path):
        # 兜底：在输出目录找最新的 mp3
        mp3s = [
            os.path.join(output_dir, f)
            for f in os.listdir(output_dir)
            if f.lower().endswith(".mp3")
        ]
        if mp3s:
            mp3_path = max(mp3s, key=os.path.getmtime)

    return {
        "path": os.path.abspath(mp3_path),
        "title": info.get("title", ""),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration": info.get("duration") or 0,
        "size": os.path.getsize(mp3_path) if os.path.exists(mp3_path) else 0,
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    url = sys.argv[1].strip()
    output_dir = sys.argv[2].strip() if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    if not re.match(r"^https?://", url):
        sys.exit("请输入 http/https 开头的链接")

    print(f"开始提取: {url}")
    result = extract_mp3(url, output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n✅ MP3 地址: {result['path']}")


if __name__ == "__main__":
    main()
