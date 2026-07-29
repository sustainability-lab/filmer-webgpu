#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")/.." && pwd)"
source_video="${FILMER_CAPTURE_SOURCE:-$project_root/docs/walkthrough/actual-app-capture.mp4}"
source_audio="${FILMER_VOICE_SOURCE:-$project_root/docs/walkthrough/filmer-gemini-indian-male.wav}"
captions="$project_root/public/walkthrough/filmer-web-walkthrough.vtt"
output_video="$project_root/public/walkthrough/filmer-web-walkthrough.mp4"
poster="$project_root/public/walkthrough/filmer-web-walkthrough-poster.jpg"

if [[ ! -f "$source_video" ]]; then
  echo "Missing real app capture: $source_video" >&2
  echo "Run npm run record:walkthrough, then preserve the resulting capture at that path." >&2
  exit 1
fi

if [[ ! -f "$source_audio" ]]; then
  echo "Missing Gemini narration: $source_audio" >&2
  exit 1
fi

mkdir -p "$(dirname "$output_video")"

# The source is a single genuine production-app run:
#   0–7 s       selections and launch
#   7–89.5 s    real NOAA download and local ONNX inference
#   89.5–98.2 s actual map, animation, zoom, pan, and reset
# Only the waiting period is accelerated. No screen is simulated or replaced.
ffmpeg -hide_banner -loglevel error -y \
  -i "$source_video" \
  -i "$source_audio" \
  -filter_complex \
  "[0:v]trim=start=0:end=7,setpts=PTS-STARTPTS[v0]; \
   [0:v]trim=start=7:end=89.5,setpts=(PTS-STARTPTS)/4.8[v1]; \
   [0:v]trim=start=89.5:end=98.2,setpts=(PTS-STARTPTS)/0.5[v2]; \
   [v0][v1][v2]concat=n=3:v=1:a=0, \
   tpad=stop_mode=clone:stop_duration=8.41, \
   fps=30,format=yuv420p,fade=t=in:st=0:d=0.35,fade=t=out:st=49.4:d=0.6[v]; \
   [1:a]loudnorm=I=-16:LRA=7:TP=-1.5,apad=pad_dur=0.6,afade=t=out:st=49.4:d=0.5[a]" \
  -map "[v]" \
  -map "[a]" \
  -map_metadata -1 \
  -c:v libx264 \
  -preset slow \
  -crf 18 \
  -movflags +faststart \
  -c:a aac \
  -b:a 160k \
  -ar 48000 \
  -t 50 \
  "$output_video"

ffmpeg -hide_banner -loglevel error -y \
  -ss 32 \
  -i "$output_video" \
  -frames:v 1 \
  -q:v 2 \
  "$poster"

ffprobe -v error \
  -show_entries format=duration,size \
  -show_entries stream=codec_name,width,height,r_frame_rate,sample_rate,channels \
  -of json \
  "$output_video"

echo "Built genuine-app walkthrough: $output_video"
echo "Captions: $captions"
