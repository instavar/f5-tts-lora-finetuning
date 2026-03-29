# F5-TTS LoRA Training Log

## Run 2: F5TTS_v1_Base + FEMALE_01 (production)

**Date:** 2026-03-29
**Status:** Complete — training + inference validated

### Base model

| Field | Value |
|---|---|
| Model | `F5TTS_v1_Base` (DiT, dim=1024, depth=22, heads=16, ff_mult=2) |
| Checkpoint | `hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors` |
| SHA256 prefix | `670900fd14e6c458` |
| Source | HuggingFace Hub, auto-downloaded via `cached_path` |
| Config diffs vs v0 | `text_mask_padding=True`, no `pe_attn_head` (v0 has `pe_attn_head=1`) |

### Dataset

| Field | Value |
|---|---|
| Name | IMDA NSC FEMALE_01 (Singaporean English) |
| Tokenizer | pinyin |
| Train samples | 10,850 |
| Data path | `data/female01_v1_lora_pinyin/` → symlink to F5-TTS-Plus train split |
| Split source | `female01_pinyin_split_summary.json` (seed=20251115, 90/5/5) |

### LoRA config

| Parameter | Value |
|---|---|
| Rank (r) | 16 |
| Alpha | 16 (scaling = 1.0) |
| Dropout | 0.0 |
| Target modules | `to_q`, `to_k`, `to_v`, `to_out.0` |
| Bias | none |
| Trainable params | 2,883,584 / 339,980,388 (0.85%) |
| EMA | Disabled (Option 3 — no EMA for LoRA) |

### Training hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 1e-4 |
| Optimizer | AdamW (fused) |
| Epochs | 1 |
| Warmup updates | 100 |
| Batch size | 3200 frames/GPU |
| Max samples/batch | 64 |
| Grad accumulation | 1 |
| Max grad norm | 1.0 |
| Total updates | ~1,250 |

### Hardware

| Field | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3090 Ti (24 GB) |
| Host | desktop_tailscale |
| Python | 3.10.19 (miniconda3, env: f5tts) |
| PEFT | 0.18.1 |
| F5-TTS | 1.1.18 (instavar fork, commit dc1b18a) |

### Outputs

| Artifact | Path | Size |
|---|---|---|
| Adapter (last) | `ckpts/female01_v1_lora/lora_last/adapter_model.safetensors` | 12 MB |
| Training state | `ckpts/female01_v1_lora/lora_last/training_state.pt` | 23 MB |
| Adapter config | `ckpts/female01_v1_lora/lora_last/adapter_config.json` | 1.1 KB |
| Checkpoints | `lora_250`, `lora_500`, `lora_750`, `lora_1000`, `lora_1250`, `lora_last` | |
| Adapter SHA256 prefix | `9cae13fa69da810a` | |
| Inference test | `tests/lora_v1_infer_test.wav` | 482 KB (10.3s @ 24kHz) |

### Exact training command

```bash
cd /mnt/work/chee-wei-jie/voice-models/f5-tts-lora-finetuning

WANDB_MODE=disabled ~/miniconda3/envs/f5tts/bin/python -m accelerate.commands.launch \
  src/f5_tts/train/finetune_cli.py \
  --exp_name F5TTS_v1_Base \
  --dataset_name female01_v1_lora \
  --finetune \
  --lora \
  --lora_rank 16 \
  --lora_alpha 16 \
  --learning_rate 1e-4 \
  --epochs 1 \
  --num_warmup_updates 100 \
  --save_per_updates 250 \
  --last_per_updates 100 \
  --batch_size_per_gpu 3200 \
  --batch_size_type frame \
  --max_samples 64 \
  --tokenizer pinyin
```

### Exact inference command

```python
from f5_tts.api import F5TTS

tts = F5TTS(
    model="F5TTS_v1_Base",
    lora_path="ckpts/female01_v1_lora/lora_last",
    device="cuda",
)

wav, sr, spec = tts.infer(
    ref_file="<reference_audio.wav>",
    ref_text="",
    gen_text="Text to synthesize.",
    file_wave="output.wav",
    seed=42,
)
```

---

## Run 1: F5TTS_Base (v0) + FEMALE_01 (smoke test)

**Date:** 2026-03-29
**Status:** Complete — used for initial pipeline validation only

### Base model

| Field | Value |
|---|---|
| Model | `F5TTS_Base` (DiT, dim=1024, depth=22, heads=16, ff_mult=2) |
| Checkpoint | `F5-TTS-Plus/ckpts/F5TTS_Base/model_1200000.safetensors` (local copy) |
| SHA256 prefix | `4180310f91d592ce` |
| Config diffs vs v1 | `text_mask_padding=False`, `pe_attn_head=1` |

### Training

Same LoRA config and hyperparameters as Run 2. Used `--exp_name F5TTS_Base` with `--pretrain` pointing to local v0 checkpoint.

### Outputs

| Artifact | Path | Size |
|---|---|---|
| Checkpoint dir | `ckpts/female01_v0_lora_smoke/` | |
| Adapter SHA256 prefix | `53af7ba85605c115` | |
| Inference test | `tests/lora_infer_test.wav` | 344 KB (7.3s @ 24kHz) |

### Purpose

Pipeline validation only. Confirmed: PEFT injection on plain `nn.Module`, optimizer with filtered trainable params, `save_pretrained`/`from_pretrained` round-trip, `merge_and_unload` at inference. All worked first try.

---

## Checkpoint directory layout

```
ckpts/
├── female01_v0_lora_smoke/        # Run 1 (smoke test, v0 base)
│   ├── pretrained_model_1200000.safetensors   # v0 base (copied)
│   ├── pretrained_model_1250000.safetensors   # v1 base (auto-downloaded, unused)
│   ├── lora_250/ ... lora_1250/               # intermediate checkpoints
│   └── lora_last/                             # final adapter
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── training_state.pt
│
├── female01_v1_lora/              # Run 2 (production, v1 base)
│   ├── pretrained_model_1250000.safetensors   # v1 base (copied from HF cache)
│   ├── lora_250/ ... lora_1250/               # intermediate checkpoints
│   └── lora_last/                             # final adapter
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── training_state.pt
│
tests/
├── lora_infer_test.wav            # Run 1 inference output
└── lora_v1_infer_test.wav         # Run 2 inference output
```
