FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel@sha256:e96c6896ecfbb50d89c87bf94110206ef444f27268c5f72201eb29fba9c90331

USER root

ARG DEBIAN_FRONTEND=noninteractive

ARG SOURCE_REVISION=unknown

LABEL org.opencontainers.image.source="https://github.com/instavar/f5-tts-lora-finetuning"
LABEL org.opencontainers.image.revision="$SOURCE_REVISION"

RUN set -x \
    && apt-get update \
    && apt-get -y install wget curl man git less openssl libssl-dev unzip unar build-essential aria2 tmux vim \
    && apt-get install -y openssh-server sox libsox-fmt-all libsox-fmt-mp3 libsndfile1-dev ffmpeg \
    && apt-get install -y librdmacm1 libibumad3 librdmacm-dev libibverbs1 libibverbs-dev ibverbs-utils ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
    
WORKDIR /workspace/F5-TTS

COPY . .

RUN test -f src/f5_tts/train/lora_resume_contract.py \
    && test -f src/third_party/BigVGAN/bigvgan.py \
    && pip install -e . --no-cache-dir \
    && python -c "from pathlib import Path; import f5_tts; import f5_tts.train.lora_resume_contract; assert Path(f5_tts.__file__).resolve().is_relative_to(Path('/workspace/F5-TTS'))"

ENV SHELL=/bin/bash

VOLUME /root/.cache/huggingface/hub/

EXPOSE 7860
