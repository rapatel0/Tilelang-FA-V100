#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="${1:?usage: run_verifier_version_matrix.sh OUTPUT_DIR}"
mkdir -p "${OUTPUT}"
for version in 0.1.8 0.1.9; do
    env_dir="${OUTPUT}/venv-${version}"
    result_dir="${OUTPUT}/tilelang-${version}"
    python3 -m venv --system-site-packages "${env_dir}"
    "${env_dir}/bin/pip" install --disable-pip-version-check --no-deps --force-reinstall "tilelang==${version}"
    "${env_dir}/bin/pip" install --disable-pip-version-check --no-deps -e "${ROOT}"
    "${env_dir}/bin/python" "${ROOT}/tools/run_paged_verify_case.py" --output "${result_dir}/cases"
    "${env_dir}/bin/python" "${ROOT}/tools/export_paged_verify.py" \
        --output "${result_dir}/export" --no-causal --num-pages 1024 --max-blocks 1024
done
python3 - "${OUTPUT}" <<'PY'
import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
versions = ("0.1.8", "0.1.9")
reports = {
    version: json.loads((root / f"tilelang-{version}" / "cases" / "results.json").read_text())
    for version in versions
}
case_hashes = {
    version: {case["case"]: case["output_sha256"] for case in reports[version]["cases"]}
    for version in versions
}
if case_hashes[versions[0]] != case_hashes[versions[1]]:
    raise SystemExit(f"TileLang output mismatch: {case_hashes}")
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
artifacts = {}
for version in versions:
    export = root / f"tilelang-{version}" / "export"
    artifacts[version] = {
        name: sha(export / name)
        for name in ("partial.cu", "partial.ptx", "partial.sass", "combine.cu", "combine.ptx", "combine.sass")
    }
result = {
    "schema": 1,
    "output_parity": "bit-identical",
    "case_hashes": case_hashes,
    "artifact_sha256": artifacts,
}
path = root / "matrix.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(path)
PY
