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
    require(dockerfile, "SOURCE_REVISION", label="Dockerfile")
    reject(dockerfile, "git clone https://github.com/SWivid/F5-TTS", label="Dockerfile")
    require(workflow, "submodules: recursive", label="Docker workflow")
    require(workflow, "push: ${{ github.event_name == 'push' }}", label="Docker workflow")
    require(workflow, "python scripts/check_docker_contract.py", label="Docker workflow")


if __name__ == "__main__":
    main()
