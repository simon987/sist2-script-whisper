#!/bin/bash

# Whisper.cpp
git clone https://github.com/ggerganov/whisper.cpp
(
  cd whisper.cpp || exit

  make

  bash ./models/download-ggml-model.sh large

  if [ ! -f models/ggml-large-q4_1.bin ]; then
    ./quantize models/ggml-large.bin models/ggml-large-q4_1.bin q4_1 >/dev/null
  fi

  bash ./models/download-ggml-model.sh base

  if [ ! -f models/ggml-base-q4_1.bin ]; then
    ./quantize models/ggml-base.bin models/ggml-base-q4_1.bin q4_1 >/dev/null
  fi

  bash ./models/download-ggml-model.sh small

  if [ ! -f models/ggml-small-q4_1.bin ]; then
    ./quantize models/ggml-small.bin models/ggml-small-q4_1.bin q4_1 >/dev/null
  fi
)

# ffmpeg
if ! [ -x "$(command -v git)" ]; then
  echo "Installing ffmpeg"
  apt update -y
  apt install -y ffmpeg
fi

# Python requirements
pip install -r requirements.txt