#!/usr/bin/env python3
"""Fail when the built image drifts from the qualified owned-source stack."""

from importlib.metadata import version
from pathlib import Path

import f5_tts.train.lora_resume_contract as resume_contract


EXPECTED_VERSIONS = {
    "accelerate": "1.11.0",
    "datasets": "4.4.1",
    "dill": "0.4.0",
    "fsspec": "2025.10.0",
    "multiprocess": "0.70.18",
    "numpy": "1.26.4",
    "peft": "0.18.1",
    "pyarrow": "22.0.0",
    "torch": "2.9.0+cu128",
    "torchaudio": "2.9.0+cu128",
    "torchcodec": "0.8.1",
    "transformers": "4.57.1",
}
EXPECTED_SOURCE_ROOT = Path("/workspace/F5-TTS")


def main() -> None:
    actual_versions = {name: version(name) for name in EXPECTED_VERSIONS}
    if actual_versions != EXPECTED_VERSIONS:
        raise SystemExit(f"container dependency drift: expected={EXPECTED_VERSIONS!r} actual={actual_versions!r}")

    contract_path = Path(resume_contract.__file__).resolve()
    if not contract_path.is_relative_to(EXPECTED_SOURCE_ROOT):
        raise SystemExit(f"container imported the resume contract outside the owned checkout: {contract_path}")


if __name__ == "__main__":
    main()
