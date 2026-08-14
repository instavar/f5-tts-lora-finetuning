from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(source: str, needle: str, *, label: str) -> None:
    if needle not in source:
        raise SystemExit(f"{label} is missing: {needle}")


def reject(source: str, needle: str, *, label: str) -> None:
    if needle in source:
        raise SystemExit(f"{label} contains forbidden text: {needle}")


def main() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/publish-docker-image.yaml").read_text(encoding="utf-8")
    require(
        dockerfile,
        "FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel@sha256:",
        label="Dockerfile",
    )
    require(dockerfile, "COPY . .", label="Dockerfile")
    require(
        dockerfile,
        "test -f src/f5_tts/train/lora_resume_contract.py",
        label="Dockerfile",
    )
    require(
        dockerfile,
        "test -f src/third_party/BigVGAN/bigvgan.py",
        label="Dockerfile",
    )
    require(
        dockerfile,
        "python scripts/check_container_runtime.py",
        label="Dockerfile",
    )
    require(dockerfile, "--constraint docker-constraints.txt", label="Dockerfile")
    constraints = (ROOT / "docker-constraints.txt").read_text(encoding="utf-8")
    for dependency in (
        "datasets==4.4.1",
        "dill==0.4.0",
        "fsspec==2025.10.0",
        "multiprocess==0.70.18",
        "numpy==1.26.4",
        "torch==2.9.0",
        "torchaudio==2.9.0",
        "torchcodec==0.8.1",
        "transformers==4.57.1",
        "peft==0.18.1",
        "accelerate==1.11.0",
        "pyarrow==22.0.0",
    ):
        require(constraints, dependency, label="Docker constraints")
    runtime_check = (ROOT / "scripts/check_container_runtime.py").read_text(
        encoding="utf-8"
    )
    require(
        runtime_check,
        "import f5_tts.train.lora_resume_contract as resume_contract",
        label="Container runtime check",
    )
    require(
        runtime_check,
        'EXPECTED_SOURCE_ROOT = Path("/workspace/F5-TTS")',
        label="Container runtime check",
    )
    require(dockerfile, "SOURCE_REVISION", label="Dockerfile")
    reject(dockerfile, "git clone https://github.com/SWivid/F5-TTS", label="Dockerfile")
    require(workflow, "submodules: recursive", label="Docker workflow")
    require(workflow, "push: ${{ github.event_name == 'push' }}", label="Docker workflow")
    require(workflow, "python scripts/check_docker_contract.py", label="Docker workflow")


if __name__ == "__main__":
    main()
