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
from pathlib import Path, PurePosixPath
from typing import Any


REPO_ROOT = Path(__file__).parents[1]
UPSTREAM_BASE_REVISION = "82fc4fe622fe36047d1dff99b550e6018181ea11"


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


def _safe_name(value: str) -> str:
    path = Path(value)
    if not value or value in {".", ".."} or path.is_absolute() or len(path.parts) != 1:
        raise ValueError("SELECTED_ADAPTER_NAME must be one safe child directory")
    return value


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
            "schema_version": "1.0.0",
            "status": "passed",
            "companion_revision": revision,
            "upstream_base_revision": UPSTREAM_BASE_REVISION,
            "base_checkpoint_sha256": _sha256(base),
            "base_checkpoint_bytes": base.stat().st_size,
            "persistent_package_root": str(persistent_package_root),
            "persistence_probe": persistence_probe,
            "corpus_audit": audit,
            "generation_rows": len(rows),
            "dataset_lineage": lineage,
        },
    )


def _train() -> None:
    _verify_dataset_lineage()
    work = _work()
    output = work / "train" / "output"
    command = [
        "accelerate",
        "launch",
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
        os.environ.get("LORA_RANK", "16"),
        "--lora_alpha",
        os.environ.get("LORA_ALPHA", "16"),
        "--learning_rate",
        os.environ.get("LEARNING_RATE", "1e-4"),
        "--epochs",
        os.environ.get("EPOCHS", "20"),
        "--num_warmup_updates",
        os.environ.get("NUM_WARMUP_UPDATES", "200"),
        "--save_per_updates",
        os.environ.get("SAVE_PER_UPDATES", "500"),
        "--last_per_updates",
        os.environ.get("LAST_PER_UPDATES", "100"),
        "--batch_size_per_gpu",
        os.environ.get("BATCH_SIZE_PER_GPU", "3200"),
        "--batch_size_type",
        os.environ.get("BATCH_SIZE_TYPE", "frame"),
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
        os.environ.get("SMOKE_TEXT", "A held-out sentence verifies adapter reload."),
        "--output_dir",
        str(output.parent),
        "--output_file",
        output.name,
    ]
    if os.environ.get("VOCAB_FILE"):
        command.extend(["--vocab_file", os.environ["VOCAB_FILE"]])
    _run(command)
    if not output.is_file() or output.stat().st_size == 0:
        raise ValueError("fresh-process adapter inference did not produce audio")


def _evaluate() -> None:
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
    preflight = json.loads((work / "preflight" / "preflight.json").read_text(encoding="utf-8"))
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
            "schema_version": "1.0.0",
            "backend_id": "f5-tts-lora-pytorch",
            "external_base_checkpoint": {
                "sha256": preflight["base_checkpoint_sha256"],
                "bytes": preflight["base_checkpoint_bytes"],
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
    actions = {"preflight": _preflight, "train": _train, "infer": _infer, "evaluate": _evaluate, "package": _package}
    if stage not in actions:
        raise ValueError(f"unknown lifecycle stage: {stage}")
    actions[stage]()
    if stage in {"preflight", "train"}:
        _verify_dataset_lineage()
    _write_json(
        Path(os.environ["INSTAVAR_VOICE_STAGE_RESULT"]), {"schema_version": "1.0.0", "stage": stage, "status": "passed"}
    )


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
