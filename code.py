
conda activate moss-transcribe-diarize
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
vllm serve /home/administator/workspace/hub/MOSS-Transcribe-Diarize \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 9005 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 1 \
  --limit-mm-per-prompt '{"audio":1}' \
  --max-model-len 65536 \
  --max-num-batched-tokens 32768


curl http://localhost:9005/v1/audio/transcriptions \
  -F 'model=/home/administator/workspace/hub/MOSS-Transcribe-Diarize' \
  -F 'file=@/home/administator/workspace/mp3/录音.m4a' \
  -F 'response_format=json' \
  -F 'temperature=0' \
  -o /home/administator/workspace/mp3/录音.txt

curl http://localhost:9005/v1/audio/transcriptions \
  -F 'model=/home/administator/workspace/hub/MOSS-Transcribe-Diarize' \
  -F 'file=@/home/administator/workspace/mp3/录音.m4a' \
  -F 'response_format=json' \
  -F 'temperature=0' \
  -o /home/administator/workspace/mp3/录音.json

ffmpeg -y \
  -i "/home/administator/workspace/mp3/录音.mp3" \
  -ac 1 \
  -ar 16000 \
  -c:a libmp3lame \
  -b:a 128k \
  "/home/administator/workspace/mp3/converted.mp3"



curl http://localhost:9005/v1/audio/transcriptions \
  -F 'model=/home/administator/workspace/hub/MOSS-Transcribe-Diarize' \
  -F 'file=@/mnt/e/mp3/特朗普为何“大撒币”？中期选举究竟有多重要？【现实版纸牌屋】.mp3' \
  -F 'response_format=json' \
  -F 'temperature=0' \
  -o /home/administator/workspace/trump.json



# 将音频按每段 600 秒（10 分钟）切分，输出到 /tmp/ 目录
ffmpeg -i /mnt/e/mp3/特朗普为何“大撒币”？中期选举究竟有多重要？【现实版纸牌屋】.mp3 -f segment -segment_time 120 -c copy /home/administator/workspace/mp3/data_tlp/audio_part_%03d.mp3

for f in /home/administator/workspace/mp3/data_tlp/audio_part_*.mp3; do
  curl http://localhost:9005/v1/audio/transcriptions \
    -F 'model=/home/administator/workspace/hub/MOSS-Transcribe-Diarize' \
    -F "file=@$f" \
    -F 'response_format=json' \
    -F 'temperature=0' \
    -o "/home/administator/workspace/mp3/$(basename "$f").json"
done

cat /home/administator/workspace/mp3/data_tlp/audio_part_*.mp3.json | jq -r '.text' > /home/administator/workspace/mp3/full_transcript.txt
