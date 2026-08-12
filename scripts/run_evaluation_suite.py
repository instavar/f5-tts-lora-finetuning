#!/usr/bin/env python3
"""Run a frozen Instavar Voice generation plan with one loaded F5-TTS adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import chunk_text, infer_process, preprocess_ref_audio_text
from f5_tts.model.utils import seed_everything


IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="F5TTS_Base")
    parser.add_argument("--model-checkpoint", required=True)
    parser.add_argument("--vocab-file", default="")
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--reference-audio", required=True)
    parser.add_argument("--reference-text", required=True)
    parser.add_argument("--generation-plan", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--runtime-id", default="pytorch")
    parser.add_argument("--artifact-set-id")
    parser.add_argument("--artifact-set-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--nfe-step", type=int, default=32)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    parser.add_argument("--sway-sampling-coef", type=float, default=-1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--vocoder-local-path")
    parser.add_argument(
        "--allow-invalid-output",
        action="store_true",
        help="return success after recording every planned attempt even when an output is invalid",
    )
    return parser.parse_args()


def read_rows(path: Path, candidate_id: str) -> list[dict]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("schema_version") not in {"1.0.0", "1.1.0"}:
        raise ValueError("generation plan schema_version must equal 1.0.0 or 1.1.0")
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == candidate_id]
    if not rows:
        raise ValueError(f"generation plan has no rows for candidate {candidate_id!r}")
    return rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_observations(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def runtime_artifact_fields(args: argparse.Namespace) -> dict[str, str]:
    if not IDENTIFIER_RE.fullmatch(args.runtime_id):
        raise ValueError("runtime id must be a lowercase machine-readable identifier")
    if bool(args.artifact_set_id) != bool(args.artifact_set_sha256):
        raise ValueError("artifact set id and sha256 must be provided together")
    fields = {"runtime_id": args.runtime_id}
    if args.artifact_set_id:
        if not IDENTIFIER_RE.fullmatch(args.artifact_set_id):
            raise ValueError("artifact set id must be a lowercase machine-readable identifier")
        if not re.fullmatch(r"[0-9a-f]{64}", args.artifact_set_sha256):
            raise ValueError("artifact set sha256 must be a lowercase SHA-256 digest")
        fields.update(
            {
                "artifact_set_id": args.artifact_set_id,
                "artifact_set_sha256": args.artifact_set_sha256,
            }
        )
    return fields


def cross_fade(waves: list[np.ndarray], sample_rate: int, seconds: float = 0.15) -> np.ndarray:
    combined = waves[0]
    for wave in waves[1:]:
        count = min(int(seconds * sample_rate), len(combined), len(wave))
        if count <= 0:
            combined = np.concatenate([combined, wave])
            continue
        overlap = combined[-count:] * np.linspace(1, 0, count) + wave[:count] * np.linspace(0, 1, count)
        combined = np.concatenate([combined[:-count], overlap, wave[count:]])
    return combined


def infer_sequential(engine: F5TTS, args: argparse.Namespace, text: str, seed: int) -> tuple[np.ndarray, int, int]:
    """Avoid F5's concurrent multi-chunk path, which is unsafe on one loaded model."""
    ref_file, ref_text = preprocess_ref_audio_text(args.reference_audio, args.reference_text, show_info=lambda _: None)
    ref_audio, ref_sample_rate = torchaudio.load(ref_file)
    ref_seconds = ref_audio.shape[-1] / ref_sample_rate
    max_chars = int(len(ref_text.encode("utf-8")) / ref_seconds * (22 - ref_seconds) * args.speed)
    chunks = chunk_text(text, max_chars=max_chars)
    if not chunks:
        raise ValueError("generated text produced no inference chunks")

    seed_everything(seed)
    waves: list[np.ndarray] = []
    sample_rate = 0
    for chunk in chunks:
        wave, sample_rate, _ = infer_process(
            ref_file,
            ref_text,
            chunk,
            engine.ema_model,
            engine.vocoder,
            engine.mel_spec_type,
            show_info=lambda _: None,
            progress=None,
            cross_fade_duration=0,
            nfe_step=args.nfe_step,
            cfg_strength=args.cfg_strength,
            sway_sampling_coef=args.sway_sampling_coef,
            speed=args.speed,
            device=engine.device,
        )
        if wave is None or not len(wave):
            raise RuntimeError("F5-TTS returned an empty audio chunk")
        waves.append(wave)
    return cross_fade(waves, sample_rate), sample_rate, len(chunks)


def main() -> int:
    args = parse_args()
    device_family = args.device.split(":", 1)[0].casefold()
    uses_cuda = device_family == "cuda"
    uses_mps = device_family == "mps"
    if uses_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA runtime was requested but is unavailable")
    if uses_mps and not torch.backends.mps.is_available():
        raise RuntimeError("MPS runtime was requested but is unavailable")
    artifact_fields = runtime_artifact_fields(args)
    rows = read_rows(args.generation_plan, args.candidate_id)
    engine = F5TTS(
        model=args.model,
        ckpt_file=args.model_checkpoint,
        vocab_file=args.vocab_file,
        device=args.device,
        vocoder_local_path=args.vocoder_local_path,
        lora_path=args.adapter,
    )

    observations: list[dict] = []
    for row in rows:
        output = args.output_dir / row["expected_audio_path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        if uses_cuda:
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        elif uses_mps:
            torch.mps.synchronize()
        started = time.perf_counter()
        observation = {
            "observation_schema_version": "1.0.0",
            "sample_id": row["sample_id"],
            "candidate_id": row["candidate_id"],
            "prompt_id": row["prompt_id"],
            "category": row["category"],
            "seed": row["seed"],
            "requested_text": row["text"],
            "valid": False,
            "runtime": "f5_tts_pytorch_cuda_adapter",
            **artifact_fields,
            "instruction_applied": False,
        }
        try:
            wave, sample_rate, chunk_count = infer_sequential(engine, args, row["text"], int(row["seed"]))
            sf.write(output, wave, sample_rate)
            if uses_cuda:
                torch.cuda.synchronize()
            elif uses_mps:
                torch.mps.synchronize()
            elapsed = time.perf_counter() - started
            info = sf.info(output)
            observation.update(
                {
                    "valid": info.frames > 0,
                    "audio_path": str(output),
                    "audio_sha256": sha256(output),
                    **({"audio_duration_seconds": float(info.duration)} if info.duration > 0 else {}),
                    "generation_seconds": elapsed,
                    **({"peak_memory_bytes": int(torch.cuda.max_memory_allocated())} if uses_cuda else {}),
                    "inference_chunk_count": chunk_count,
                    "instruction_note": (
                        "F5-TTS has no separate instruction input in this path; the requested text is unchanged."
                        if row.get("instruction")
                        else None
                    ),
                }
            )
        except Exception as error:
            if uses_cuda:
                torch.cuda.synchronize()
            elif uses_mps:
                torch.mps.synchronize()
            observation.update(
                {
                    "generation_seconds": time.perf_counter() - started,
                    **({"peak_memory_bytes": int(torch.cuda.max_memory_allocated())} if uses_cuda else {}),
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
        observations.append(observation)
        write_observations(args.output_dir / "generation-observations.json", observations)
    return 0 if args.allow_invalid_output or all(row["valid"] for row in observations) else 1


if __name__ == "__main__":
    raise SystemExit(main())
