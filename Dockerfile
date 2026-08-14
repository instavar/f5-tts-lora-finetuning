FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel@sha256:97ec2a667dd7560c615bf50a95b2fb85a673ae233a55da1706e8e04e6d6d518e

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
    && pip install -e . --no-cache-dir --constraint docker-constraints.txt \
    && python -c "from importlib.metadata import version; from pathlib import Path; import f5_tts.train.lora_resume_contract as contract; expected = {'torch': '2.9.0', 'torchaudio': '2.9.0', 'torchcodec': '0.8.1', 'transformers': '4.57.1', 'peft': '0.18.1', 'accelerate': '1.11.0'}; assert {name: version(name) for name in expected} == expected; assert Path(contract.__file__).resolve().is_relative_to(Path('/workspace/F5-TTS'))"

ENV SHELL=/bin/bash

VOLUME /root/.cache/huggingface/hub/

EXPOSE 7860
