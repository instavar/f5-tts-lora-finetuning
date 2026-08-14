"""Immutable initial-adapter publication and identity checks for F5 LoRA."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable


RECEIPT_NAME = "initial-adapter-receipt.json"
RECEIPT_SCHEMA_VERSION = "1.0.0"


class InitialAdapterError(ValueError):
    """Raised when initial LoRA adapter identity or publication is unsafe."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_identities(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    inodes: set[tuple[int, int]] = set()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise InitialAdapterError(f"Initial adapter rejects symlinks: {item}")
        if item.is_dir():
            continue
        if not item.is_file():
            raise InitialAdapterError(f"Initial adapter contains a non-regular member: {item}")
        stat = item.stat()
        inode = (stat.st_dev, stat.st_ino)
        if inode in inodes:
            raise InitialAdapterError("Initial adapter files must not share hardlinks")
        inodes.add(inode)
        files.append(
            {
                "path": item.relative_to(root).as_posix(),
                "bytes": stat.st_size,
                "sha256": sha256_file(item),
            }
        )
    return files


def _resolve_adapter(path: str | Path) -> Path:
    unresolved = Path(path).expanduser()
    if unresolved.is_symlink():
        raise InitialAdapterError(f"Initial adapter symlinks are not allowed: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_dir():
        raise InitialAdapterError(f"Initial adapter must be a directory: {resolved}")
    return resolved


def initial_adapter_identity(path: str | Path) -> dict[str, Any]:
    resolved = _resolve_adapter(path)
    files = _file_identities(resolved)
    names = {entry["path"] for entry in files}
    required = {"adapter_config.json", "adapter_model.safetensors", RECEIPT_NAME}
    if not required.issubset(names):
        missing = sorted(required - names)
        raise InitialAdapterError("Initial adapter omits required files: " + ", ".join(missing))
    if "adapter_model.bin" in names:
        raise InitialAdapterError("Initial adapter must use safe serialized weights")

    receipt = json.loads((resolved / RECEIPT_NAME).read_text(encoding="utf-8"))
    if receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        raise InitialAdapterError("Unsupported initial-adapter receipt schema")
    if receipt.get("producer_repository") != "instavar/f5-tts-lora-finetuning":
        raise InitialAdapterError("Initial-adapter receipt has the wrong producer repository")
    producer_revision = receipt.get("producer_revision")
    if (
        not isinstance(producer_revision, str)
        or len(producer_revision) != 40
        or any(value not in "0123456789abcdef" for value in producer_revision)
    ):
        raise InitialAdapterError("Initial-adapter receipt has an invalid producer revision")
    seed = receipt.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**63 - 1:
        raise InitialAdapterError("Initial-adapter receipt has an invalid seed")
    recorded_base = receipt.get("base_checkpoint")
    if not isinstance(recorded_base, str) or not Path(recorded_base).is_absolute():
        raise InitialAdapterError("Initial-adapter receipt has an invalid base checkpoint")
    expected = [entry for entry in files if entry["path"] != RECEIPT_NAME]
    if receipt.get("files") != expected:
        raise InitialAdapterError("Initial-adapter receipt does not bind the live file tree")
    config = json.loads((resolved / "adapter_config.json").read_text(encoding="utf-8"))
    config_targets = config.get("target_modules")
    if not isinstance(config_targets, list):
        raise InitialAdapterError("Initial adapter target_modules must be a list")
    actual_lora = {
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "dropout": config.get("lora_dropout"),
        "target_modules": sorted(config_targets),
    }
    receipt_lora = receipt.get("lora")
    if not isinstance(receipt_lora, dict) or actual_lora != receipt_lora:
        raise InitialAdapterError("Initial-adapter receipt LoRA configuration drift")
    return {"path": str(resolved), "files": files}


def validate_initial_adapter_config(
    path: str | Path,
    *,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Iterable[str],
) -> None:
    resolved = _resolve_adapter(path)
    config = json.loads((resolved / "adapter_config.json").read_text(encoding="utf-8"))
    expected_targets = sorted(set(target_modules))
    actual_targets = config.get("target_modules")
    if not isinstance(actual_targets, list):
        raise InitialAdapterError("Initial adapter target_modules must be a list")
    expected = {
        "r": rank,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "target_modules": expected_targets,
        "bias": "none",
    }
    actual = {
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "lora_dropout": config.get("lora_dropout"),
        "target_modules": sorted(actual_targets),
        "bias": config.get("bias"),
    }
    if actual != expected:
        raise InitialAdapterError(f"Initial adapter LoRA configuration drift: expected={expected!r} actual={actual!r}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_initial_adapter(
    model: Any,
    output: str | Path,
    *,
    producer_revision: str,
    base_checkpoint: str | Path,
    seed: int,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: Iterable[str],
) -> Path:
    if len(producer_revision) != 40 or any(value not in "0123456789abcdef" for value in producer_revision):
        raise InitialAdapterError("producer_revision must be a lowercase 40-character Git commit")
    unresolved = Path(output).expanduser()
    if not unresolved.is_absolute():
        raise InitialAdapterError("Initial adapter output must be absolute")
    if unresolved.exists() or unresolved.is_symlink():
        raise FileExistsError(f"Refusing to overwrite initial adapter: {unresolved}")
    unresolved.parent.mkdir(parents=True, exist_ok=True)
    parent = unresolved.parent.resolve(strict=True)
    target = parent / unresolved.name
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite initial adapter: {target}")

    temporary = parent / f".{unresolved.name}.{os.getpid()}.partial"
    temporary.mkdir(exist_ok=False)
    model.save_pretrained(temporary, safe_serialization=True)
    files = _file_identities(temporary)
    names = {entry["path"] for entry in files}
    if not {"adapter_config.json", "adapter_model.safetensors"}.issubset(names):
        raise InitialAdapterError("PEFT did not publish the required safe adapter files")
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "producer_repository": "instavar/f5-tts-lora-finetuning",
        "producer_revision": producer_revision,
        "base_checkpoint": str(Path(base_checkpoint).expanduser().resolve(strict=True)),
        "seed": seed,
        "lora": {
            "rank": rank,
            "alpha": alpha,
            "dropout": dropout,
            "target_modules": sorted(set(target_modules)),
        },
        "files": files,
        "evidence_boundary": (
            "These are serialized initial adapter bytes. A training process must explicitly "
            "load and content-bind this directory before using it as conditioning evidence."
        ),
    }
    receipt_path = temporary / RECEIPT_NAME
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    for item in temporary.rglob("*"):
        if item.is_file():
            with item.open("rb") as handle:
                os.fsync(handle.fileno())
    _fsync_directory(temporary)
    os.replace(temporary, target)
    _fsync_directory(parent)
    return target
