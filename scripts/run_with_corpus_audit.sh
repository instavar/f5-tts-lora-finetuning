#!/usr/bin/env bash
set -euo pipefail

evaluator_dir="${INSTAVAR_VOICE_EVAL_DIR:?Set INSTAVAR_VOICE_EVAL_DIR to a pinned instavar-voice-evaluation checkout}"
python_bin="${PYTHON:-python3}"
output="artifacts/corpus-audit.json"
group_field=""
train_seen=0
validation_seen=0
test_seen=0
audit_args=()

while [[ $# -gt 0 && "$1" != "--" ]]; do
  case "$1" in
    --split)
      [[ $# -ge 2 ]] || { echo "--split requires NAME=PATH" >&2; exit 2; }
      split_name="${2%%=*}"
      split_path="${2#*=}"
      [[ "$split_name" != "$split_path" && -n "$split_path" ]] || {
        echo "--split requires NAME=PATH" >&2
        exit 2
      }
      case "$split_name" in
        train) train_seen=1 ;;
        validation) validation_seen=1 ;;
        test) test_seen=1 ;;
        *) echo "Unsupported split: $split_name" >&2; exit 2 ;;
      esac
      audit_args+=(--split "$2")
      shift 2
      ;;
    --group-field)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--group-field requires a value" >&2; exit 2; }
      group_field="$2"
      shift 2
      ;;
    --audit-output)
      [[ $# -ge 2 && -n "$2" ]] || { echo "--audit-output requires a path" >&2; exit 2; }
      output="$2"
      shift 2
      ;;
    *)
      echo "Unknown audit option: $1" >&2
      exit 2
      ;;
  esac
done

[[ $# -gt 0 && "$1" == "--" ]] || { echo "Separate the training command with --" >&2; exit 2; }
shift
[[ $# -gt 0 ]] || { echo "Training command is required after --" >&2; exit 2; }
[[ $train_seen -eq 1 && $validation_seen -eq 1 && $test_seen -eq 1 ]] || {
  echo "Explicit train, validation, and test splits are required" >&2
  exit 2
}
[[ -n "$group_field" ]] || { echo "--group-field is required" >&2; exit 2; }
[[ -f "$evaluator_dir/main.py" ]] || { echo "Evaluator main.py not found in $evaluator_dir" >&2; exit 2; }

mkdir -p "$(dirname "$output")"
"$python_bin" "$evaluator_dir/main.py" audit-corpus \
  "${audit_args[@]}" \
  --group-field "$group_field" \
  --output "$output"

exec "$@"
