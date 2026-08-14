"""Dependency-free integrity contract for F5-TTS LoRA continuation."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from f5_tts.train.lora_initial_adapter import initial_adapter_identity


SCHEMA_VERSION = "1.0.0"
SIDECAR_NAME = "resume-contract.json"
STATE_NAME = "training-state.json"
RUNTIME_STATE_NAME = "runtime-state.pt"
OPTIMIZER_STATE_NAME = "optimizer-state.pt"
SCHEDULER_STATE_NAME = "scheduler-state.pt"
_CHECKPOINT_RE = re.compile(r"^lora_(\d+)$")


class ResumeContractError(ValueError):
    """Raised when LoRA continuation state does not match the current run."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ResumeContractError(f"Expected a regular file: {resolved}")
    stat = resolved.stat()
    return {"path": str(resolved), "sha256": sha256_file(resolved), "size": stat.st_size}


def tree_identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ResumeContractError(f"Expected a directory: {resolved}")
    root_stat = resolved.stat()
    files: list[dict[str, Any]] = []
    for item in sorted(resolved.rglob("*")):
        if item.is_symlink():
            target = item.resolve(strict=True)
            if target.is_file():
                identity = file_identity(target)
                identity["path"] = item.relative_to(resolved).as_posix()
                identity["resolved_path"] = str(target)
                identity["symlink_target"] = os.readlink(item)
                files.append(identity)
            continue
        if item.is_file():
            identity = file_identity(item)
            identity["path"] = item.relative_to(resolved).as_posix()
            files.append(identity)
    return {
        "path": str(resolved),
        "device": root_stat.st_dev,
        "inode": root_stat.st_ino,
        "files": files,
    }


def build_contract(
    *,
    output_dir: str | Path,
    base_checkpoint: str | Path,
    dataset_root: str | Path,
    optional_files: Mapping[str, str | Path | None],
    initial_adapter_dir: str | Path | None = None,
    source_files: Iterable[str | Path],
    training_config: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    output = Path(output_dir).expanduser().resolve(strict=True)
    output_stat = output.stat()
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": "lora",
        "output_dir": {"path": str(output), "device": output_stat.st_dev, "inode": output_stat.st_ino},
        "base_checkpoint": file_identity(base_checkpoint),
        "dataset": tree_identity(dataset_root),
        "optional_files": {
            name: file_identity(path) for name, path in sorted(optional_files.items()) if path is not None and str(path)
        },
        "initial_adapter": initial_adapter_identity(initial_adapter_dir) if initial_adapter_dir else None,
        "sources": [file_identity(path) for path in sorted((Path(path) for path in source_files), key=str)],
        "training_config": dict(training_config),
        "runtime": dict(runtime),
    }


def contract_digest(contract: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(contract).encode("utf-8")).hexdigest()


def require_fresh_output(output_dir: str | Path) -> None:
    output = Path(output_dir).expanduser().resolve(strict=True)
    conflicts = [
        item.name
        for item in output.iterdir()
        if item.name == "lora_last" or _CHECKPOINT_RE.fullmatch(item.name) or item.name.startswith(".lora_")
    ]
    if conflicts:
        rendered = ", ".join(sorted(conflicts)[:5])
        raise ResumeContractError(
            f"Output directory already contains LoRA checkpoint state ({rendered}); use a fresh directory or explicit resume_from"
        )


def prunable_checkpoints(output_dir: str | Path, keep_last: int) -> list[Path]:
    output = Path(output_dir).expanduser().resolve(strict=True)
    checkpoints: list[tuple[int, Path]] = []
    for item in output.iterdir():
        match = _CHECKPOINT_RE.fullmatch(item.name)
        if not match:
            continue
        if item.is_symlink():
            raise ResumeContractError(f"Refusing to prune symbolic LoRA checkpoint: {item}")
        if not item.is_dir() or item.parent != output:
            raise ResumeContractError(f"Refusing to prune unsafe LoRA checkpoint: {item}")
        ownership_marker = item / SIDECAR_NAME
        if ownership_marker.is_symlink() or not ownership_marker.is_file():
            raise ResumeContractError(f"Refusing to prune unowned LoRA checkpoint: {item}")
        checkpoints.append((int(match.group(1)), item))
    checkpoints.sort(key=lambda entry: entry[0])
    return [path for _, path in checkpoints[: max(0, len(checkpoints) - keep_last)]]


def resolve_checkpoint(checkpoint: str | Path, output_dir: str | Path) -> Path:
    raw = Path(checkpoint).expanduser()
    if raw.is_symlink():
        raise ResumeContractError(f"Checkpoint symlinks are not allowed: {raw}")
    resolved = raw.resolve(strict=True)
    output = Path(output_dir).expanduser().resolve(strict=True)
    if resolved.parent != output:
        raise ResumeContractError("Resume checkpoint must be a direct child of the configured output directory")
    if not resolved.is_dir() or not _CHECKPOINT_RE.fullmatch(resolved.name):
        raise ResumeContractError("Resume checkpoint must be an immutable lora_N directory")
    return resolved


def write_sidecar(
    checkpoint_dir: str | Path,
    *,
    contract: Mapping[str, Any],
    completed_updates: int,
    required_files: Iterable[str],
) -> Path:
    checkpoint = Path(checkpoint_dir).resolve(strict=True)
    files: dict[str, dict[str, Any]] = {}
    for name in sorted(set(required_files)):
        if Path(name).name != name:
            raise ResumeContractError(f"Checkpoint member must be a basename: {name}")
        member = checkpoint / name
        if member.is_symlink() or not member.is_file():
            raise ResumeContractError(f"Required checkpoint member is missing or unsafe: {member}")
        files[name] = {"sha256": sha256_file(member), "size": member.stat().st_size}
    payload = {
        "schema_version": SCHEMA_VERSION,
        "completed_updates": int(completed_updates),
        "contract_sha256": contract_digest(contract),
        "contract": dict(contract),
        "files": files,
    }
    target = checkpoint / SIDECAR_NAME
    temporary = checkpoint / f".{SIDECAR_NAME}.{os.getpid()}.tmp"
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    directory_fd = os.open(checkpoint, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return target


def validate_checkpoint(
    checkpoint: str | Path,
    *,
    output_dir: str | Path,
    expected_contract: Mapping[str, Any],
    trust_resume_state: bool,
    world_size: int,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if not trust_resume_state:
        raise ResumeContractError(
            "LoRA resume includes PyTorch optimizer state; set trust_resume_state=true only for state you trust"
        )
    if world_size != 1:
        raise ResumeContractError(
            "Guarded LoRA resume supports world_size=1 only because rank-local loader and RNG state are not persisted"
        )
    resolved = resolve_checkpoint(checkpoint, output_dir)
    sidecar_path = resolved / SIDECAR_NAME
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise ResumeContractError(f"Checkpoint has no safe {SIDECAR_NAME}: {resolved}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise ResumeContractError("Unsupported resume sidecar schema")
    if sidecar.get("contract_sha256") != contract_digest(expected_contract):
        raise ResumeContractError("LoRA resume contract drift detected")
    if sidecar.get("contract") != dict(expected_contract):
        raise ResumeContractError("LoRA resume contract payload does not match the current run")
    files = sidecar.get("files")
    required = {
        "adapter_config.json",
        "adapter_model.safetensors",
        "training_state.pt",
        STATE_NAME,
        RUNTIME_STATE_NAME,
    }
    if not isinstance(files, dict) or not required.issubset(files):
        raise ResumeContractError("LoRA resume sidecar does not bind every required continuation file")
    for name, identity in files.items():
        if Path(name).name != name or not isinstance(identity, dict):
            raise ResumeContractError("LoRA resume sidecar contains an unsafe file entry")
        member = resolved / name
        if member.is_symlink() or not member.is_file():
            raise ResumeContractError(f"LoRA resume member is missing or unsafe: {name}")
        if member.stat().st_size != identity.get("size") or sha256_file(member) != identity.get("sha256"):
            raise ResumeContractError(f"LoRA resume member drift detected: {name}")
    state = json.loads((resolved / STATE_NAME).read_text(encoding="utf-8"))
    completed_updates = state.get("completed_updates")
    if not isinstance(completed_updates, int) or completed_updates < 1:
        raise ResumeContractError("LoRA training state has invalid completed_updates")
    if completed_updates != sidecar.get("completed_updates"):
        raise ResumeContractError("LoRA training state and sidecar disagree on completed updates")
    if resolved.name != f"lora_{completed_updates}":
        raise ResumeContractError("LoRA checkpoint directory does not match completed-update state")
    for key in ("data_epoch", "batches_consumed_in_epoch"):
        if not isinstance(state.get(key), int) or state[key] < 0:
            raise ResumeContractError(f"LoRA training state has invalid {key}")
    target_epochs = expected_contract.get("training_config", {}).get("epochs")
    if isinstance(target_epochs, int) and state["data_epoch"] >= target_epochs:
        raise ResumeContractError("LoRA checkpoint already reached the configured training target")
    return resolved, state, sidecar


def evaluator_lora_artifact_paths(checkpoint: str | Path) -> dict[str, Path]:
    """Map one published checkpoint to evaluator 0.45 final-state roles."""
    unresolved = Path(checkpoint).expanduser()
    if unresolved.is_symlink():
        raise ResumeContractError(f"Evaluator checkpoint symlinks are not allowed: {unresolved}")
    resolved = unresolved.resolve(strict=True)
    if not resolved.is_dir() or not _CHECKPOINT_RE.fullmatch(resolved.name):
        raise ResumeContractError("Evaluator checkpoint must be an immutable lora_N directory")

    sidecar_path = resolved / SIDECAR_NAME
    if sidecar_path.is_symlink() or not sidecar_path.is_file():
        raise ResumeContractError(f"Checkpoint has no safe {SIDECAR_NAME}: {resolved}")
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if sidecar.get("schema_version") != SCHEMA_VERSION:
        raise ResumeContractError("Unsupported resume sidecar schema")
    files = sidecar.get("files")
    if not isinstance(files, dict):
        raise ResumeContractError("Resume sidecar files must be an object")

    names_by_role = {
        "model_state": "adapter_model.safetensors",
        "optimizer_state": OPTIMIZER_STATE_NAME,
        "scheduler_state": SCHEDULER_STATE_NAME,
        "trainer_state": STATE_NAME,
        "rng_state": RUNTIME_STATE_NAME,
    }
    missing = sorted(role for role, name in names_by_role.items() if name not in files)
    if missing:
        raise ResumeContractError("Evaluator checkpoint omits roles: " + ", ".join(missing))

    artifacts: dict[str, Path] = {}
    identities: set[tuple[int, int]] = set()
    for role, name in names_by_role.items():
        member = resolved / name
        identity = files[name]
        if member.is_symlink() or not member.is_file() or not isinstance(identity, dict):
            raise ResumeContractError(f"Evaluator checkpoint member is missing or unsafe: {name}")
        stat = member.stat()
        file_token = (stat.st_dev, stat.st_ino)
        if file_token in identities:
            raise ResumeContractError("Evaluator artifact roles must not share files or hardlinks")
        identities.add(file_token)
        if stat.st_size != identity.get("size") or sha256_file(member) != identity.get("sha256"):
            raise ResumeContractError(f"Evaluator checkpoint member drift detected: {name}")
        artifacts[role] = member
    return artifacts
