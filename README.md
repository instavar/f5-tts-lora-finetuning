# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

[![python](https://img.shields.io/badge/Python-3.10-brightgreen)](https://github.com/SWivid/F5-TTS)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06885-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2410.06885)
[![demo](https://img.shields.io/badge/GitHub-Demo-orange.svg)](https://swivid.github.io/F5-TTS/)
[![hfspace](https://img.shields.io/badge/🤗-HF%20Space-yellow)](https://huggingface.co/spaces/mrfakename/E2-F5-TTS)
[![msspace](https://img.shields.io/badge/🤖-MS%20Space-blue)](https://modelscope.cn/studios/AI-ModelScope/E2-F5-TTS)
[![lab](https://img.shields.io/badge/🏫-X--LANCE-grey?labelColor=lightgrey)](https://x-lance.sjtu.edu.cn/)
[![lab](https://img.shields.io/badge/🏫-SII-grey?labelColor=lightgrey)](https://www.sii.edu.cn/)
[![lab](https://img.shields.io/badge/🏫-PCL-grey?labelColor=lightgrey)](https://www.pcl.ac.cn)
<!-- <img src="https://github.com/user-attachments/assets/12d7749c-071a-427c-81bf-b87b91def670" alt="Watermark" style="width: 40px; height: auto"> -->

**F5-TTS**: Diffusion Transformer with ConvNeXt V2, faster trained and inference.

**E2 TTS**: Flat-UNet Transformer, closest reproduction from [paper](https://arxiv.org/abs/2406.18009).

**Sway Sampling**: Inference-time flow step sampling strategy, greatly improves performance

### Thanks to all the contributors !

## News
- **2025/03/12**: 🔥 F5-TTS v1 base model with better training and inference performance. [Few demo](https://swivid.github.io/F5-TTS_updates).
- **2024/10/08**: F5-TTS & E2 TTS base models on [🤗 Hugging Face](https://huggingface.co/SWivid/F5-TTS), [🤖 Model Scope](https://www.modelscope.cn/models/SWivid/F5-TTS_Emilia-ZH-EN), [🟣 Wisemodel](https://wisemodel.cn/models/SJTU_X-LANCE/F5-TTS_Emilia-ZH-EN).

## Installation

### Create a separate environment if needed

```bash
# Create a conda env with python_version>=3.10  (you could also use virtualenv)
conda create -n f5-tts python=3.11
conda activate f5-tts

# Install FFmpeg if you haven't yet
conda install ffmpeg
```

### Install PyTorch with matched device

<details>
<summary>NVIDIA GPU</summary>

> ```bash
> # Install pytorch with your CUDA version, e.g.
> pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
> 
> # And also possible previous versions, e.g.
> pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
> # etc.
> ```

</details>

<details>
<summary>AMD GPU</summary>

> ```bash
> # Install pytorch with your ROCm version (Linux only), e.g.
> pip install torch==2.5.1+rocm6.2 torchaudio==2.5.1+rocm6.2 --extra-index-url https://download.pytorch.org/whl/rocm6.2
> ```

</details>

<details>
<summary>Intel GPU</summary>

> ```bash
> # Install pytorch with your XPU version, e.g.
> # Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit must be installed
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/test/xpu
> 
> # Intel GPU support is also available through IPEX (Intel® Extension for PyTorch)
> # IPEX does not require the Intel® Deep Learning Essentials or Intel® oneAPI Base Toolkit
> # See: https://pytorch-extension.intel.com/installation?request=platform
> ```

</details>

<details>
<summary>Apple Silicon</summary>

> ```bash
> # Install the stable pytorch, e.g.
> pip install torch torchaudio
> ```

</details>

### Then you can choose one from below:

> ### 1. As a pip package (if just for inference)
> 
> ```bash
> pip install f5-tts
> ```
> 
> ### 2. Local editable (if also do training, finetuning)
> 
> ```bash
> git clone https://github.com/SWivid/F5-TTS.git
> cd F5-TTS
> # git submodule update --init --recursive  # (optional, if use bigvgan as vocoder)
> pip install -e .
> ```

### Docker usage also available
```bash
# Build from Dockerfile
docker build -t f5tts:v1 .

# Run from GitHub Container Registry
docker container run --rm -it --gpus=all --mount 'type=volume,source=f5-tts,target=/root/.cache/huggingface/hub/' -p 7860:7860 ghcr.io/swivid/f5-tts:main

# Quickstart if you want to just run the web interface (not CLI)
docker container run --rm -it --gpus=all --mount 'type=volume,source=f5-tts,target=/root/.cache/huggingface/hub/' -p 7860:7860 ghcr.io/swivid/f5-tts:main f5-tts_infer-gradio --host 0.0.0.0
```

### Runtime

Deployment solution with Triton and TensorRT-LLM.

#### Benchmark Results
Decoding on a single L20 GPU, using 26 different prompt_audio & target_text pairs, 16 NFE.

| Model               | Concurrency    | Avg Latency | RTF    | Mode            |
|---------------------|----------------|-------------|--------|-----------------|
| F5-TTS Base (Vocos) | 2              | 253 ms      | 0.0394 | Client-Server   |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.0402 | Offline TRT-LLM |
| F5-TTS Base (Vocos) | 1 (Batch_size) | -           | 0.1467 | Offline Pytorch |

See [detailed instructions](src/f5_tts/runtime/triton_trtllm/README.md) for more information.


## Inference

- In order to achieve desired performance, take a moment to read [detailed guidance](src/f5_tts/infer).
- By properly searching the keywords of problem encountered, [issues](https://github.com/SWivid/F5-TTS/issues?q=is%3Aissue) are very helpful.

### 1. Gradio App

Currently supported features:

- Basic TTS with Chunk Inference
- Multi-Style / Multi-Speaker Generation
- Voice Chat powered by Qwen2.5-3B-Instruct
- [Custom inference with more language support](src/f5_tts/infer/SHARED.md)

```bash
# Launch a Gradio app (web interface)
f5-tts_infer-gradio

# Specify the port/host
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# Launch a share link
f5-tts_infer-gradio --share
```

<details>
<summary>NVIDIA device docker compose file example</summary>

```yaml
services:
  f5-tts:
    image: ghcr.io/swivid/f5-tts:main
    ports:
      - "7860:7860"
    environment:
      GRADIO_SERVER_PORT: 7860
    entrypoint: ["f5-tts_infer-gradio", "--port", "7860", "--host", "0.0.0.0"]
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  f5-tts:
    driver: local
```

</details>

### 2. CLI Inference

```bash
# Run with flags
# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli --model F5TTS_v1_Base \
--ref_audio "provide_prompt_wav_path_here.wav" \
--ref_text "The content, subtitle or transcription of reference audio." \
--gen_text "Some text you want TTS model generate for you."

# Run with default setting. src/f5_tts/infer/examples/basic/basic.toml
f5-tts_infer-cli
# Or with your own .toml file
f5-tts_infer-cli -c custom.toml

# Multi voice. See src/f5_tts/infer/README.md
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml
```


## Training

### 1. With Hugging Face Accelerate

Refer to [training & finetuning guidance](src/f5_tts/train) for best practice.

### 2. With Gradio App

```bash
# Quick start with Gradio web interface
f5-tts_finetune-gradio
```

Read [training & finetuning guidance](src/f5_tts/train) for more instructions.


## [Evaluation](src/f5_tts/eval)

### Frozen multi-prompt adapter evaluation

Load a selected LoRA adapter once and execute every F5-TTS row in an Instavar
Voice generation plan:

```bash
python scripts/run_evaluation_suite.py \
  --model F5TTS_Base \
  --model-checkpoint /path/to/model_1200000.safetensors \
  --adapter /path/to/lora_step_1250 \
  --reference-audio /path/to/reference.wav \
  --reference-text "The exact transcript of the reference audio." \
  --generation-plan evaluation/generation-plan.json \
  --candidate-id f5-lora1250 \
  --output-dir evaluation/f5-lora1250
```

The runner records every planned attempt and uses the frozen seed per sample.
The F5 path has no separate style-instruction input, so instruction prompts are
generated from the unchanged transcript and explicitly marked as not applied.
This makes the capability difference visible instead of silently rewriting the
test.


## Development

Use pre-commit to ensure code quality (will run linters and formatters automatically):

```bash
pip install pre-commit
pre-commit install
```

When making a pull request, before each commit, run: 

```bash
pre-commit run --all-files
```

Note: Some model components have linting exceptions for E722 to accommodate tensor notation.


## Acknowledgements

- [E2-TTS](https://arxiv.org/abs/2406.18009) brilliant work, simple and effective
- [Emilia](https://arxiv.org/abs/2407.05361), [WenetSpeech4TTS](https://arxiv.org/abs/2406.05763), [LibriTTS](https://arxiv.org/abs/1904.02882), [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) valuable datasets
- [lucidrains](https://github.com/lucidrains) initial CFM structure with also [bfs18](https://github.com/bfs18) for discussion
- [SD3](https://arxiv.org/abs/2403.03206) & [Hugging Face diffusers](https://github.com/huggingface/diffusers) DiT and MMDiT code structure
- [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as ODE solver, [Vocos](https://huggingface.co/charactr/vocos-mel-24khz) and [BigVGAN](https://github.com/NVIDIA/BigVGAN) as vocoder
- [FunASR](https://github.com/modelscope/FunASR), [faster-whisper](https://github.com/SYSTRAN/faster-whisper), [UniSpeech](https://github.com/microsoft/UniSpeech), [SpeechMOS](https://github.com/tarepan/SpeechMOS) for evaluation tools
- [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner) for speech edit test
- [mrfakename](https://x.com/realmrfakename) huggingface space demo ~
- [f5-tts-mlx](https://github.com/lucasnewman/f5-tts-mlx/tree/main) Implementation with MLX framework by [Lucas Newman](https://github.com/lucasnewman)
- [F5-TTS-ONNX](https://github.com/DakeQQ/F5-TTS-ONNX) ONNX Runtime version by [DakeQQ](https://github.com/DakeQQ)
- [Yuekai Zhang](https://github.com/yuekaizhang) Triton and TensorRT-LLM support ~

## LoRA Fine-tuning (instavar fork)

This fork adds **LoRA (Low-Rank Adaptation)** support to F5-TTS via [PEFT](https://github.com/huggingface/peft). LoRA freezes the base model and trains small adapter matrices (~0.9% of parameters at rank 16), reducing VRAM usage and producing tiny checkpoint files (~6 MB vs ~2.7 GB for full fine-tuning).

### Quick Start

```bash
# Audit JSONL manifests for the same corpus and split assignment, then train.
export INSTAVAR_VOICE_EVAL_DIR=/path/to/instavar-voice-evaluation
scripts/run_with_corpus_audit.sh \
  --split train=data/audit/train.jsonl \
  --split validation=data/audit/validation.jsonl \
  --split test=data/audit/test.jsonl \
  --group-field recording_id \
  -- accelerate launch src/f5_tts/train/finetune_cli.py \
  --exp_name F5TTS_v1_Base \
  --dataset_name my_speaker \
  --finetune \
  --lora \
  --lora_rank 16 \
  --lora_alpha 16 \
  --learning_rate 1e-4 \
  --epochs 20 \
  --num_warmup_updates 200 \
  --save_per_updates 500 \
  --last_per_updates 100 \
  --batch_size_per_gpu 3200 \
  --batch_size_type frame
```

### Executable Instavar Voice lifecycle

[`instavar-voice-backend.json`](instavar-voice-backend.json) binds LoRA training
and PyTorch merged-adapter inference to a real five-stage lifecycle. The trainer
now accepts `--checkpoint_path`, so lifecycle artifacts are written under the
unique work directory instead of the repository's shared `ckpts/` tree. The
wrapper audits grouped splits, selects one exact adapter directory, reloads it
in a fresh process with the same optional vocabulary file, uses the sequential
multi-chunk frozen evaluator, and packages the adapter plus preflight, smoke,
evaluation, experiment, and plan evidence.

Validate the recipe with evaluator merge
`d63ab559a8e0592bd373f9b51421040b540fb2b7` and use an empty work directory
outside the checkout. A passed lifecycle establishes execution and artifact
lineage. It does not establish that the adapted voice is perceptually better or
that inherited Triton, MLX, or ONNX runtimes reproduce the PyTorch result.

### LoRA CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--lora` | `False` | Enable LoRA fine-tuning mode |
| `--lora_rank` | `16` | LoRA rank (r). Higher = more capacity, more params |
| `--lora_alpha` | `16` | LoRA alpha. Scaling factor = alpha/rank |
| `--lora_dropout` | `0.0` | Dropout on LoRA layers (0.0 recommended for small datasets) |
| `--lora_target_modules` | `to_q to_k to_v to_out.0` | Which layers to apply LoRA to |

### Target Modules

The default targets the attention Q/K/V/O projections in the DiT transformer blocks. You can also include the feedforward layers:

```bash
# Attention + FFN (broader adaptation, ~1.7% of params at r=16)
--lora_target_modules to_q to_k to_v to_out.0 ff.ff.0.0 ff.ff.2
```

### Inference with LoRA

LoRA adapters are merged into the base model at load time (zero inference overhead):

```bash
# CLI inference
f5-tts_infer-cli \
  --lora_path ckpts/my_speaker/lora_last \
  -r ref_audio.wav \
  -s "Reference text." \
  -t "Text to synthesize."

# Python API
from f5_tts.api import F5TTS
tts = F5TTS(lora_path="ckpts/my_speaker/lora_last")
wav, sr, spec = tts.infer(ref_file="ref.wav", ref_text="...", gen_text="...")
```

### LoRA Config also works in TOML config files

```toml
# basic.toml
lora_path = "ckpts/my_speaker/lora_last"
```

### Key Design Decisions

- **No EMA in LoRA mode**: The base model already has well-trained EMA weights from pre-training. LoRA adds a small delta. This matches community practice for diffusion LoRA (Kohya, diffusers).
- **Separate adapter checkpoints**: LoRA saves `adapter_config.json` + `adapter_model.safetensors` (~6 MB) instead of full model checkpoints (~2.7 GB).
- **Merge at inference**: `merge_and_unload()` bakes LoRA weights into the base model, producing a standard `nn.Module` with zero inference overhead.
- **Resumable training**: Optimizer and scheduler states are saved alongside LoRA adapters for training resumption.

## Citation
If our work and codebase is useful for you, please cite as:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.

## Instavar Voice conformance

[`instavar-voice-capabilities.json`](instavar-voice-capabilities.json) separates the Instavar LoRA path from inherited upstream training, evaluation, and runtime surfaces. It keeps Triton TensorRT-LLM, MLX, and ONNX visible without claiming adapter equivalence that has not been reproduced. CI validates the manifest against the pinned public [Instavar Voice evaluation contract](https://github.com/instavar/instavar-voice-evaluation).
