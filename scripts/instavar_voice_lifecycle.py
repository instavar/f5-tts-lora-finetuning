#!/usr/bin/env python3
"""Execute F5-TTS LoRA through the Instavar Voice lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import wave
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).parents[1]
UPSTREAM_BASE_REVISION = "82fc4fe622fe36047d1dff99b550e6018181ea11"
PREFLIGHT_SCHEMA_VERSION = "1.3.0"
PACKAGE_SCHEMA_VERSION = "1.2.0"
DEFAULT_SMOKE_TEXT = "A held-out sentence verifies adapter reload."
DEFAULT_SMOKE_SEED = 42
SUPPORTED_MODELS = {"E2TTS_Base", "F5TTS_Base", "F5TTS_v1_Base"}
PACKAGE_MEMBER_NAMES = {
    "dataset-lineage.json",
    "evaluation-bundle.tar",
    "experiment-manifest.json",
    "generation-plan.json",
    "preflight.json",
    "selected-adapter.tar",
    "smoke-candidate.wav",
}


def _path(name: str, *, directory: bool = False) -> Path:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    unresolved = Path(value).expanduser()
    if unresolved.is_symlink():
        raise FileNotFoundError(f"{name} is a symlink: {unresolved}")
    path = unresolved.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise FileNotFoundError(f"{name} is missing or unsafe: {path}")
    return path


def _work() -> Path:
    return _path("INSTAVAR_VOICE_WORK_DIR", directory=True)


def _persistent_package_root() -> Path:
    root = _path("PERSISTED_PACKAGE_ROOT", directory=True)
    work = _work()
    repository = REPO_ROOT.resolve()
    if root == work or root.is_relative_to(work):
        raise ValueError("PERSISTED_PACKAGE_ROOT must be outside the lifecycle work directory")
    if root == repository or root.is_relative_to(repository):
        raise ValueError("PERSISTED_PACKAGE_ROOT must be outside the repository checkout")
    return root


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _probe_persistent_package_root(root: Path) -> dict[str, Any]:
    probe_path: Path | None = None
    linked_path: Path | None = None
    linked_created = False
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=root,
            prefix=".instavar-voice-persistence-probe.",
            suffix=".partial",
            delete=False,
        ) as probe:
            probe_path = Path(probe.name)
            probe.write(b"instavar-voice-persistence-probe-v1\n")
            probe.flush()
            os.fsync(probe.fileno())
        linked_path = probe_path.with_suffix(".linked")
        os.link(probe_path, linked_path)
        linked_created = True
        _fsync_directory(root)
        if linked_path.read_bytes() != probe_path.read_bytes():
            raise ValueError("persistent package root failed its atomic publication probe")
        identity = root.stat()
        return {
            "writable": True,
            "atomic_hard_link": True,
            "device": identity.st_dev,
            "inode": identity.st_ino,
        }
    except OSError as error:
        raise ValueError(f"PERSISTED_PACKAGE_ROOT cannot publish an atomic package: {error}") from error
    finally:
        if linked_path is not None and linked_created:
            linked_path.unlink(missing_ok=True)
        if probe_path is not None:
            probe_path.unlink(missing_ok=True)


def _locked_persistent_package_root(preflight: dict[str, Any]) -> Path:
    root = _persistent_package_root()
    recorded_root = preflight.get("persistent_package_root")
    recorded_device = preflight.get("persistence_probe", {}).get("device")
    recorded_inode = preflight.get("persistence_probe", {}).get("inode")
    identity = root.stat()
    if recorded_root != str(root) or recorded_device != identity.st_dev or recorded_inode != identity.st_ino:
        raise ValueError("PERSISTED_PACKAGE_ROOT changed after preflight")
    return root


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _tree_sha256(root: Path) -> str:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"tree root is missing or unsafe: {root}")
    digest = hashlib.sha256()
    count = 0
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"tree contains a symlink: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(path.stat().st_size.to_bytes(8, "big"))
            digest.update(bytes.fromhex(_sha256(path)))
            count += 1
        elif not path.is_dir():
            raise ValueError(f"tree contains an unsupported entry: {path}")
    if count == 0:
        raise ValueError("tree contains no files")
    return digest.hexdigest()


def _external_tree_identity(root: Path) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"external dependency tree is missing or unsafe: {root}")
    files = [path for path in root.rglob("*") if path.is_file()]
    return {
        "sha256": _tree_sha256(root),
        "bytes": sum(path.stat().st_size for path in files),
        "files": len(files),
    }


def _safe_name(value: str) -> str:
    path = Path(value)
    if not value or value in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError("SELECTED_ADAPTER_NAME must be one safe child directory")
    return value


def _training_config() -> dict[str, str]:
    return {
        "batch_size_per_gpu": os.environ.get("BATCH_SIZE_PER_GPU", "3200"),
        "batch_size_type": os.environ.get("BATCH_SIZE_TYPE", "frame"),
        "epochs": os.environ.get("EPOCHS", "20"),
        "last_per_updates": os.environ.get("LAST_PER_UPDATES", "100"),
        "learning_rate": os.environ.get("LEARNING_RATE", "1e-4"),
        "lora_alpha": os.environ.get("LORA_ALPHA", "16"),
        "lora_rank": os.environ.get("LORA_RANK", "16"),
        "num_warmup_updates": os.environ.get("NUM_WARMUP_UPDATES", "200"),
        "save_per_updates": os.environ.get("SAVE_PER_UPDATES", "500"),
        "seed": str(_seed(os.environ.get("TRAINING_SEED", "666"), label="TRAINING_SEED")),
    }


def _seed(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer from 0 through 2^63 - 1")
    try:
        seed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be an integer from 0 through 2^63 - 1") from error
    if str(seed) != str(value).strip() or not 0 <= seed <= 2**63 - 1:
        raise ValueError(f"{label} must be an integer from 0 through 2^63 - 1")
    return seed


def _smoke_seed() -> int:
    return _seed(os.environ.get("SMOKE_SEED", str(DEFAULT_SMOKE_SEED)), label="SMOKE_SEED")


def _smoke_text() -> str:
    value = os.environ.get("SMOKE_TEXT", DEFAULT_SMOKE_TEXT)
    if not value:
        raise ValueError("SMOKE_TEXT must not be empty")
    return value


def _bound_lifecycle_inputs() -> dict[str, Any]:
    model = os.environ.get("MODEL", "F5TTS_v1_Base")
    if model not in SUPPORTED_MODELS:
        raise ValueError("MODEL is not supported by the F5-TTS lifecycle")
    reference_text = os.environ.get("REFERENCE_TEXT", "")
    if not reference_text:
        raise ValueError("REFERENCE_TEXT is required")
    smoke_text = _smoke_text()
    candidate_id = os.environ.get("CANDIDATE_ID", "").strip()
    dataset_name = os.environ.get("DATASET_NAME", "").strip()
    if not candidate_id or not dataset_name:
        raise ValueError("CANDIDATE_ID and DATASET_NAME are required")
    vocab_file = _path("VOCAB_FILE") if os.environ.get("VOCAB_FILE", "").strip() else None
    return {
        "files": {
            "base_checkpoint": _external_file_identity(_path("BASE_MODEL_CHECKPOINT")),
            "dataset_lineage": _external_file_identity(_path("DATASET_LINEAGE")),
            "experiment_manifest": _external_file_identity(_path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST")),
            "generation_plan": _external_file_identity(_path("GENERATION_PLAN")),
            "raw_test": _external_file_identity(_path("RAW_TEST_JSONL")),
            "raw_train": _external_file_identity(_path("RAW_TRAIN_JSONL")),
            "raw_validation": _external_file_identity(_path("RAW_VALIDATION_JSONL")),
            "reference_audio": _external_file_identity(_path("REFERENCE_AUDIO")),
            "vocabulary": _external_file_identity(vocab_file) if vocab_file is not None else None,
            "vocoder": _external_tree_identity(_path("VOCODER_DIR", directory=True)),
        },
        "values": {
            "candidate_id": candidate_id,
            "corpus_group_field": os.environ.get("CORPUS_GROUP_FIELD", ""),
            "dataset_name": dataset_name,
            "model": model,
            "reference_text_sha256": _text_sha256(reference_text),
            "selected_adapter_name": _safe_name(os.environ.get("SELECTED_ADAPTER_NAME", "")),
            "smoke_seed": _smoke_seed(),
            "smoke_text_sha256": _text_sha256(smoke_text),
            "training_config": _training_config(),
        },
    }


def _verified_preflight() -> dict[str, Any]:
    path = _work() / "preflight" / "preflight.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("lifecycle has no safe preflight receipt")
    preflight = json.loads(path.read_text(encoding="utf-8"))
    if preflight.get("schema_version") != PREFLIGHT_SCHEMA_VERSION or preflight.get("status") != "passed":
        raise ValueError(f"lifecycle requires a passed schema {PREFLIGHT_SCHEMA_VERSION} preflight")
    if preflight.get("companion_revision") != _git_head():
        raise ValueError("companion revision changed after preflight")
    if preflight.get("bound_inputs") != _bound_lifecycle_inputs():
        raise ValueError("lifecycle inputs or controls changed after preflight")
    return preflight


def _run(command: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(command, cwd=REPO_ROOT, capture_output=capture, text=capture, check=False)
    if result.returncode != 0:
        detail = (result.stderr or "").strip() if capture else ""
        raise RuntimeError(f"command failed with exit code {result.returncode}: {command[0]}: {detail}")
    return (result.stdout or "").strip() if capture else ""


def _git_head() -> str:
    return _run(["git", "rev-parse", "HEAD"], capture=True)


def _git_clean() -> bool:
    return not _run(["git", "status", "--porcelain=v1", "--untracked-files=all"], capture=True)


def _prepared_dataset() -> Path:
    name = os.environ["DATASET_NAME"].strip()
    candidate = Path(name)
    if not name or name in {".", ".."} or candidate.is_absolute() or len(candidate.parts) != 1:
        raise ValueError("DATASET_NAME must be one safe dataset identifier")
    return _path_from_value(REPO_ROOT / "data" / f"{name}_pinyin", directory=True, label="prepared dataset")


def _path_from_value(value: Path, *, directory: bool, label: str) -> Path:
    if value.is_symlink():
        raise FileNotFoundError(f"{label} is a symlink: {value}")
    path = value.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        raise FileNotFoundError(f"{label} is missing: {path}")
    return path


def _verify_dataset_lineage() -> dict[str, Any]:
    from instavar_voice_lab.lineage import verify_dataset_lineage

    document = json.loads(_path("DATASET_LINEAGE").read_text(encoding="utf-8"))
    return verify_dataset_lineage(
        document,
        producer_revision=_git_head(),
        inputs={
            "raw_train": (_path("RAW_TRAIN_JSONL"), "file"),
            "raw_validation": (_path("RAW_VALIDATION_JSONL"), "file"),
            "raw_test": (_path("RAW_TEST_JSONL"), "file"),
        },
        outputs={"prepared_dataset": (_prepared_dataset(), "tree")},
    )


def _archive(source: Path, destination: Path, *, arcname: str) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"archive source must be a non-symlink directory: {source}")
    count = 0
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"archive source contains a symlink: {path}")
        if path.is_file():
            count += 1
        elif not path.is_dir():
            raise ValueError(f"archive source contains an unsupported entry: {path}")
    if count == 0:
        raise ValueError("archive source contains no files")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(destination, "w") as archive:
        archive.add(source, arcname=arcname, recursive=True)


def _verify_persisted_package(path: Path, expected_sha256: str) -> None:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"persisted package is missing, empty, or unsafe: {path}")
    actual_sha256 = _sha256(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"persisted package hash mismatch: expected {expected_sha256}, got {actual_sha256}")


def _persist_package(source: Path, root: Path) -> dict[str, Any]:
    if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"package source is missing, empty, or unsafe: {source}")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"persistent package root is missing or unsafe: {root}")
    package_sha256 = _sha256(source)
    destination = root / f"f5-tts-lora-package-sha256-{package_sha256}.tar"
    reused_existing = destination.exists() or destination.is_symlink()
    if reused_existing:
        _verify_persisted_package(destination, package_sha256)
    else:
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=root,
                prefix=f".{destination.name}.",
                suffix=".partial",
                delete=False,
            ) as target:
                temporary_path = Path(target.name)
                with source.open("rb") as package:
                    shutil.copyfileobj(package, target, length=1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            _verify_persisted_package(temporary_path, package_sha256)
            try:
                os.link(temporary_path, destination)
            except FileExistsError:
                reused_existing = True
            else:
                _fsync_directory(root)
            _verify_persisted_package(destination, package_sha256)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    return {
        "schema_version": "1.0.0",
        "adaptation_mode": "lora",
        "package_sha256": package_sha256,
        "package_bytes": source.stat().st_size,
        "persisted_path": str(destination),
        "reused_existing": reused_existing,
    }


def _extract(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("adapter archive is empty")
        seen: set[tuple[str, ...]] = set()
        for member in members:
            parts = PurePosixPath(member.name).parts
            target = (destination / member.name).resolve()
            if (
                not parts
                or parts[0] != "adapter"
                or ".." in parts
                or not target.is_relative_to(destination.resolve())
                or member.issym()
                or member.islnk()
                or not (member.isfile() or member.isdir())
                or parts in seen
            ):
                raise ValueError(f"unsafe adapter archive member: {member.name}")
            seen.add(parts)
        archive.extractall(destination, members=members, filter="data")
    adapter = destination / "adapter"
    if not adapter.is_dir() or not any(path.is_file() for path in adapter.rglob("*")):
        raise ValueError("adapter archive did not contain a non-empty adapter root")
    return adapter


def _extract_package(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(source, "r") as archive:
        members = archive.getmembers()
        if not members:
            raise ValueError("lifecycle package is empty")
        seen: set[tuple[str, ...]] = set()
        for member in members:
            parts = PurePosixPath(member.name).parts
            target = (destination / member.name).resolve()
            is_root_directory = member.isdir() and parts == ("package",)
            is_bound_file = (
                member.isfile() and len(parts) == 2 and parts[1] in PACKAGE_MEMBER_NAMES | {"package-manifest.json"}
            )
            if (
                not parts
                or parts[0] != "package"
                or ".." in parts
                or not target.is_relative_to(destination.resolve())
                or member.issym()
                or member.islnk()
                or not (is_root_directory or is_bound_file)
                or parts in seen
            ):
                raise ValueError(f"unsafe lifecycle package member: {member.name}")
            seen.add(parts)
        archive.extractall(destination, members=members, filter="data")
    package = destination / "package"
    if package.is_symlink() or not package.is_dir():
        raise ValueError("lifecycle archive did not contain a safe package root")
    return package


def _manifest_file_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = manifest.get("files")
    if not isinstance(rows, list):
        raise ValueError("package manifest files must be a list")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("package manifest contains an invalid file row")
        name = row.get("path")
        if not isinstance(name, str) or Path(name).name != name or name in result:
            raise ValueError("package manifest contains an unsafe or duplicate file path")
        sha256 = row.get("sha256")
        size = row.get("bytes")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
            or type(size) is not int
            or size < 1
        ):
            raise ValueError(f"package manifest contains an invalid identity for {name}")
        result[name] = row
    if set(result) != PACKAGE_MEMBER_NAMES:
        raise ValueError("package manifest does not bind the exact required file set")
    return result


def _external_file_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"external dependency is missing, empty, or unsafe: {path}")
    return {"sha256": _sha256(path), "bytes": path.stat().st_size}


def _verify_external_identity(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, dict) or _external_file_identity(path) != expected:
        raise ValueError(f"{label} does not match the lifecycle package")


def _verify_external_tree_identity(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, dict) or _external_tree_identity(path) != expected:
        raise ValueError(f"{label} does not match the lifecycle package")


def _verify_package_contents(
    package: Path,
    *,
    base_checkpoint: Path,
    reference_audio: Path,
    vocab_file: Path | None,
    vocoder_dir: Path,
) -> dict[str, Any]:
    manifest_path = package / "package-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("lifecycle package has no safe package-manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PACKAGE_SCHEMA_VERSION:
        raise ValueError("unsupported lifecycle package schema")
    if manifest.get("backend_id") != "f5-tts-lora-pytorch":
        raise ValueError("lifecycle package backend does not match F5-TTS LoRA")
    files = _manifest_file_map(manifest)
    actual_files = {
        path.relative_to(package).as_posix() for path in package.rglob("*") if path.is_file() and path != manifest_path
    }
    if actual_files != set(files):
        raise ValueError("lifecycle package contains unbound or missing files")
    for name, identity in files.items():
        path = package / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"lifecycle package member is missing or unsafe: {name}")
        if path.stat().st_size != identity["bytes"] or _sha256(path) != identity["sha256"]:
            raise ValueError(f"lifecycle package member drift detected: {name}")
    dependencies = manifest.get("external_dependencies")
    if not isinstance(dependencies, dict) or set(dependencies) != {
        "base_checkpoint",
        "reference_audio",
        "vocabulary",
        "vocoder",
    }:
        raise ValueError("lifecycle package has an invalid external dependency contract")
    _verify_external_identity(base_checkpoint, dependencies["base_checkpoint"], "base checkpoint")
    _verify_external_identity(reference_audio, dependencies["reference_audio"], "reference audio")
    expected_vocab = dependencies["vocabulary"]
    if expected_vocab is None:
        if vocab_file is not None:
            raise ValueError("a vocabulary file was supplied but the lifecycle package declares none")
    else:
        if vocab_file is None:
            raise ValueError("the lifecycle package requires an external vocabulary file")
        _verify_external_identity(vocab_file, expected_vocab, "vocabulary file")
    _verify_external_tree_identity(vocoder_dir, dependencies["vocoder"], "vocoder directory")
    inference = manifest.get("inference_contract")
    if not isinstance(inference, dict) or set(inference) != {
        "reference_text_sha256",
        "seed",
        "smoke_text_sha256",
    }:
        raise ValueError("lifecycle package has an invalid inference contract")
    for name in ("reference_text_sha256", "smoke_text_sha256"):
        value = inference[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("lifecycle package has an invalid inference contract")
    _seed(inference["seed"], label="package inference seed")
    model = manifest.get("model")
    revision = manifest.get("companion_revision")
    if (
        model not in SUPPORTED_MODELS
        or not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("lifecycle package has invalid model or companion revision provenance")
    return manifest


def _probe_wav(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size == 0:
        raise ValueError("restored inference did not produce a safe non-empty WAV")
    try:
        with wave.open(str(path), "rb") as audio:
            frames = audio.getnframes()
            sample_rate = audio.getframerate()
            channels = audio.getnchannels()
            sample_width = audio.getsampwidth()
    except (EOFError, wave.Error) as error:
        raise ValueError(f"restored inference produced an invalid WAV: {error}") from error
    if frames < 1 or sample_rate < 1 or channels < 1 or sample_width < 1:
        raise ValueError("restored inference produced an empty or invalid WAV")
    return {
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "frames": frames,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "sample_width_bytes": sample_width,
    }


def _restore() -> None:
    source = _path("PERSISTED_PACKAGE_PATH")
    expected_sha256 = os.environ.get("EXPECTED_PACKAGE_SHA256", "").strip()
    if (
        len(expected_sha256) != 64
        or any(character not in "0123456789abcdef" for character in expected_sha256)
        or _sha256(source) != expected_sha256
    ):
        raise ValueError("persisted lifecycle package does not match EXPECTED_PACKAGE_SHA256")
    base_checkpoint = _path("BASE_MODEL_CHECKPOINT")
    vocab_file = _path("VOCAB_FILE") if os.environ.get("VOCAB_FILE", "").strip() else None
    vocoder_dir = _path("VOCODER_DIR", directory=True)
    reference_audio = _path("REFERENCE_AUDIO")
    reference_text = os.environ.get("REFERENCE_TEXT", "")
    if not reference_text:
        raise ValueError("REFERENCE_TEXT is required")
    smoke_text = _smoke_text()
    destination_value = os.environ.get("RESTORE_OUTPUT_DIR", "").strip()
    if not destination_value:
        raise ValueError("RESTORE_OUTPUT_DIR is required")
    destination = Path(destination_value).expanduser()
    if not destination.is_absolute() or not destination.name:
        raise ValueError("RESTORE_OUTPUT_DIR must be an absolute child path")
    if destination.is_symlink() or destination.exists():
        raise ValueError("RESTORE_OUTPUT_DIR must not already exist")
    raw_parent = destination.parent
    if raw_parent.is_symlink():
        raise ValueError("RESTORE_OUTPUT_DIR parent is missing or unsafe")
    parent = raw_parent.resolve(strict=True)
    if not parent.is_dir():
        raise ValueError("RESTORE_OUTPUT_DIR parent is missing or unsafe")
    destination = parent / destination.name
    staging = Path(tempfile.mkdtemp(dir=parent, prefix=f".{destination.name}.partial."))
    try:
        package = _extract_package(source, staging / "archive")
        manifest = _verify_package_contents(
            package,
            base_checkpoint=base_checkpoint,
            vocab_file=vocab_file,
            vocoder_dir=vocoder_dir,
            reference_audio=reference_audio,
        )
        inference = manifest["inference_contract"]
        if _text_sha256(reference_text) != inference["reference_text_sha256"]:
            raise ValueError("REFERENCE_TEXT does not match the lifecycle package")
        if _text_sha256(smoke_text) != inference["smoke_text_sha256"]:
            raise ValueError("SMOKE_TEXT does not match the lifecycle package")
        if os.environ.get("SMOKE_SEED", "").strip() and _smoke_seed() != inference["seed"]:
            raise ValueError("SMOKE_SEED does not match the lifecycle package")
        adapter = _extract(package / "selected-adapter.tar", staging / "restored-adapter")
        output = staging / "restored-smoke.wav"
        command = [
            sys.executable,
            "-m",
            "f5_tts.infer.infer_cli",
            "--model",
            manifest["model"],
            "--ckpt_file",
            str(base_checkpoint),
            "--lora_path",
            str(adapter),
            "--ref_audio",
            str(reference_audio),
            "--ref_text",
            reference_text,
            "--gen_text",
            smoke_text,
            "--seed",
            str(inference["seed"]),
            "--output_dir",
            str(staging),
            "--output_file",
            output.name,
            "--load_vocoder_from_local",
            "--vocoder_local_path",
            str(vocoder_dir),
        ]
        if vocab_file is not None:
            command.extend(["--vocab_file", str(vocab_file)])
        _run(command)
        wav = _probe_wav(output)
        _write_json(
            staging / "restore-receipt.json",
            {
                "schema_version": "1.1.0",
                "status": "passed",
                "backend_id": manifest["backend_id"],
                "model": manifest["model"],
                "package_sha256": expected_sha256,
                "package_manifest_sha256": _sha256(package / "package-manifest.json"),
                "base_checkpoint": _external_file_identity(base_checkpoint),
                "vocabulary": _external_file_identity(vocab_file) if vocab_file is not None else None,
                "vocoder": _external_tree_identity(vocoder_dir),
                "adapter_tree_sha256": _tree_sha256(adapter),
                "reference_audio": _external_file_identity(reference_audio),
                "reference_text_sha256": _text_sha256(reference_text),
                "smoke_text_sha256": _text_sha256(command[command.index("--gen_text") + 1]),
                "seed": inference["seed"],
                "restored_wav": wav,
                "evidence_boundary": "This receipt proves package verification and a fresh-process PyTorch smoke on the declared host inputs. It does not prove remote backup, cross-runtime equivalence, perceptual quality, distribution rights, or production readiness.",
            },
        )
        os.replace(staging, destination)
        _fsync_directory(parent)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _preflight() -> None:
    from instavar_voice_lab.corpus import audit_corpus

    experiment = json.loads(_path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST").read_text(encoding="utf-8"))
    revision = _git_head()
    if not _git_clean():
        raise ValueError("companion repository must be clean; use a work directory outside the checkout")
    if experiment.get("backend", {}).get("instavar_revision") != revision:
        raise ValueError("experiment backend.instavar_revision does not match the F5 checkout")
    if experiment.get("backend", {}).get("upstream_revision") != UPSTREAM_BASE_REVISION:
        raise ValueError("experiment backend.upstream_revision does not match the pinned F5 upstream base")
    lineage = _verify_dataset_lineage()
    splits = {
        "train": _path("RAW_TRAIN_JSONL"),
        "validation": _path("RAW_VALIDATION_JSONL"),
        "test": _path("RAW_TEST_JSONL"),
    }
    audit = audit_corpus(splits, group_field=os.environ.get("CORPUS_GROUP_FIELD") or None)
    if audit["status"] != "passed":
        raise ValueError("corpus audit failed: " + "; ".join(audit["errors"]))
    base = _path("BASE_MODEL_CHECKPOINT")
    bound_inputs = _bound_lifecycle_inputs()
    persistent_package_root = _persistent_package_root()
    persistence_probe = _probe_persistent_package_root(persistent_package_root)
    _path("REFERENCE_AUDIO")
    plan = json.loads(_path("GENERATION_PLAN").read_text(encoding="utf-8"))
    rows = [row for row in plan.get("samples", []) if row.get("candidate_id") == os.environ["CANDIDATE_ID"]]
    if plan.get("schema_version") not in {"1.0.0", "1.1.0"} or not rows:
        raise ValueError("GENERATION_PLAN must be schema 1.0.0 or 1.1.0 and contain CANDIDATE_ID rows")
    _safe_name(os.environ["SELECTED_ADAPTER_NAME"])
    _write_json(
        _work() / "preflight" / "preflight.json",
        {
            "schema_version": PREFLIGHT_SCHEMA_VERSION,
            "status": "passed",
            "companion_revision": revision,
            "upstream_base_revision": UPSTREAM_BASE_REVISION,
            "model": bound_inputs["values"]["model"],
            "base_checkpoint_sha256": _sha256(base),
            "base_checkpoint_bytes": base.stat().st_size,
            "vocabulary": bound_inputs["files"]["vocabulary"],
            "bound_inputs": bound_inputs,
            "persistent_package_root": str(persistent_package_root),
            "persistence_probe": persistence_probe,
            "corpus_audit": audit,
            "generation_rows": len(rows),
            "dataset_lineage": lineage,
        },
    )


def _train() -> None:
    _verified_preflight()
    _verify_dataset_lineage()
    work = _work()
    output = work / "train" / "output"
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "src/f5_tts/train/finetune_cli.py",
        "--exp_name",
        os.environ.get("MODEL", "F5TTS_v1_Base"),
        "--dataset_name",
        os.environ["DATASET_NAME"],
        "--finetune",
        "--pretrain",
        os.environ["BASE_MODEL_CHECKPOINT"],
        "--lora",
        "--lora_rank",
        _training_config()["lora_rank"],
        "--lora_alpha",
        _training_config()["lora_alpha"],
        "--learning_rate",
        _training_config()["learning_rate"],
        "--epochs",
        _training_config()["epochs"],
        "--seed",
        _training_config()["seed"],
        "--num_warmup_updates",
        _training_config()["num_warmup_updates"],
        "--save_per_updates",
        _training_config()["save_per_updates"],
        "--last_per_updates",
        _training_config()["last_per_updates"],
        "--batch_size_per_gpu",
        _training_config()["batch_size_per_gpu"],
        "--batch_size_type",
        _training_config()["batch_size_type"],
        "--checkpoint_path",
        str(output),
    ]
    _run(command)
    _archive(
        output / _safe_name(os.environ["SELECTED_ADAPTER_NAME"]),
        work / "train" / "selected-adapter.tar",
        arcname="adapter",
    )


def _infer() -> None:
    _verified_preflight()
    work = _work()
    adapter = _extract(work / "train" / "selected-adapter.tar", work / "infer" / "reload")
    output = work / "infer" / "candidate.wav"
    command = [
        sys.executable,
        "-m",
        "f5_tts.infer.infer_cli",
        "--model",
        os.environ.get("MODEL", "F5TTS_v1_Base"),
        "--ckpt_file",
        os.environ["BASE_MODEL_CHECKPOINT"],
        "--lora_path",
        str(adapter),
        "--ref_audio",
        os.environ["REFERENCE_AUDIO"],
        "--ref_text",
        os.environ["REFERENCE_TEXT"],
        "--gen_text",
        _smoke_text(),
        "--seed",
        str(_smoke_seed()),
        "--output_dir",
        str(output.parent),
        "--output_file",
        output.name,
        "--load_vocoder_from_local",
        "--vocoder_local_path",
        str(_path("VOCODER_DIR", directory=True)),
    ]
    if os.environ.get("VOCAB_FILE"):
        command.extend(["--vocab_file", os.environ["VOCAB_FILE"]])
    _run(command)
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("fresh-process adapter inference did not produce audio")


def _evaluate() -> None:
    _verified_preflight()
    work = _work()
    adapter = _extract(work / "train" / "selected-adapter.tar", work / "evaluate" / "reload")
    output = work / "evaluate" / "output"
    command = [
        sys.executable,
        "scripts/run_evaluation_suite.py",
        "--model",
        os.environ.get("MODEL", "F5TTS_v1_Base"),
        "--model-checkpoint",
        os.environ["BASE_MODEL_CHECKPOINT"],
        "--adapter",
        str(adapter),
        "--reference-audio",
        os.environ["REFERENCE_AUDIO"],
        "--reference-text",
        os.environ["REFERENCE_TEXT"],
        "--generation-plan",
        os.environ["GENERATION_PLAN"],
        "--candidate-id",
        os.environ["CANDIDATE_ID"],
        "--output-dir",
        str(output),
        "--allow-invalid-output",
        "--vocoder-local-path",
        str(_path("VOCODER_DIR", directory=True)),
    ]
    if os.environ.get("VOCAB_FILE"):
        command.extend(["--vocab-file", os.environ["VOCAB_FILE"]])
    _run(command)
    raw_observations = output / "generation-observations.json"
    receipt = output / "generation-attempt-receipt.json"
    bound_observations = output / "objective-observations.json"
    plan = _path("GENERATION_PLAN")
    producer_revision = _git_head()
    _run(
        [
            sys.executable,
            "-m",
            "instavar_voice_lab.cli",
            "build-generation-attempt-receipt",
            str(raw_observations),
            "--plan",
            str(plan),
            "--audio-base-dir",
            str(output),
            "--producer-name",
            "f5-evaluation-runner",
            "--producer-revision",
            producer_revision,
            "--output",
            str(receipt),
        ]
    )
    _run(
        [
            sys.executable,
            "-m",
            "instavar_voice_lab.cli",
            "apply-generation-attempt-receipt",
            str(raw_observations),
            str(receipt),
            "--plan",
            str(plan),
            "--audio-base-dir",
            str(output),
            "--output",
            str(bound_observations),
        ]
    )
    _archive(output, work / "evaluate" / "evaluation-bundle.tar", arcname="evaluation")


def _package() -> None:
    work = _work()
    preflight = _verified_preflight()
    revision = preflight["companion_revision"]
    model = preflight.get("model")
    if model not in SUPPORTED_MODELS:
        raise ValueError("preflight contains an unsupported F5-TTS model")
    vocab_file = _path("VOCAB_FILE") if os.environ.get("VOCAB_FILE", "").strip() else None
    expected_vocab = preflight.get("vocabulary")
    if expected_vocab is None:
        if vocab_file is not None:
            raise ValueError("VOCAB_FILE was supplied after a no-vocabulary preflight")
    else:
        if vocab_file is None:
            raise ValueError("the preflight requires VOCAB_FILE")
        _verify_external_identity(vocab_file, expected_vocab, "vocabulary file")
    staging = work / "package" / "staging"
    staging.mkdir(parents=True, exist_ok=False)
    sources = {
        "selected-adapter.tar": work / "train" / "selected-adapter.tar",
        "evaluation-bundle.tar": work / "evaluate" / "evaluation-bundle.tar",
        "preflight.json": work / "preflight" / "preflight.json",
        "smoke-candidate.wav": work / "infer" / "candidate.wav",
        "experiment-manifest.json": _path("INSTAVAR_VOICE_EXPERIMENT_MANIFEST"),
        "generation-plan.json": _path("GENERATION_PLAN"),
        "dataset-lineage.json": _path("DATASET_LINEAGE"),
    }
    for name, source in sources.items():
        if source.is_symlink() or not source.is_file() or source.stat().st_size == 0:
            raise ValueError(f"package source is missing, empty, or unsafe: {source}")
        shutil.copyfile(source, staging / name)
    files = [
        {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in sorted(staging.iterdir())
        if path.is_file()
    ]
    _write_json(
        staging / "package-manifest.json",
        {
            "schema_version": PACKAGE_SCHEMA_VERSION,
            "backend_id": "f5-tts-lora-pytorch",
            "model": model,
            "companion_revision": revision,
            "external_dependencies": {
                "base_checkpoint": {
                    "sha256": preflight["base_checkpoint_sha256"],
                    "bytes": preflight["base_checkpoint_bytes"],
                },
                "reference_audio": preflight["bound_inputs"]["files"]["reference_audio"],
                "vocabulary": expected_vocab,
                "vocoder": preflight["bound_inputs"]["files"]["vocoder"],
            },
            "inference_contract": {
                "reference_text_sha256": preflight["bound_inputs"]["values"]["reference_text_sha256"],
                "seed": preflight["bound_inputs"]["values"]["smoke_seed"],
                "smoke_text_sha256": preflight["bound_inputs"]["values"]["smoke_text_sha256"],
            },
            "files": files,
            "evidence_boundary": "The adapter and evidence completed the lifecycle; the base checkpoint, perceptual quality, and distribution rights remain separate dependencies and gates.",
        },
    )
    package = work / "package" / "adapter-package.tar"
    _archive(staging, package, arcname="package")
    receipt = _persist_package(package, _locked_persistent_package_root(preflight))
    _write_json(work / "package" / "persisted-package.json", receipt)


def run(stage: str) -> None:
    actions = {
        "preflight": _preflight,
        "train": _train,
        "infer": _infer,
        "evaluate": _evaluate,
        "package": _package,
        "restore": _restore,
    }
    if stage not in actions:
        raise ValueError(f"unknown lifecycle stage: {stage}")
    stage_result_value = os.environ.get("INSTAVAR_VOICE_STAGE_RESULT", "").strip()
    if not stage_result_value:
        raise ValueError("INSTAVAR_VOICE_STAGE_RESULT is required")
    actions[stage]()
    if stage in {"preflight", "train"}:
        _verify_dataset_lineage()
    _write_json(Path(stage_result_value), {"schema_version": "1.0.0", "stage": stage, "status": "passed"})


def main(argv: list[str] | None = None) -> int:
    values = sys.argv[1:] if argv is None else argv
    if len(values) != 1:
        print("usage: instavar_voice_lifecycle.py STAGE", file=sys.stderr)
        return 2
    try:
        run(values[0])
    except (KeyError, OSError, RuntimeError, ValueError, json.JSONDecodeError, tarfile.TarError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
