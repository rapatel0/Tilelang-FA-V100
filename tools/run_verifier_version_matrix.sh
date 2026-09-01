#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="${1:?usage: run_verifier_version_matrix.sh OUTPUT_DIR}"
mkdir -p "${OUTPUT}"
if [ -z "${TILELANG_FA_SOURCE_COMMIT:-}" ]; then
    TILELANG_FA_SOURCE_COMMIT="$(git -C "${ROOT}" rev-parse HEAD)"
fi
if [ -z "${TILELANG_FA_SOURCE_REPOSITORY:-}" ]; then
    branch="$(git -C "${ROOT}" rev-parse --abbrev-ref HEAD)"
    remote="$(git -C "${ROOT}" config --get "branch.${branch}.remote" || true)"
    [ -n "${remote}" ] || remote=origin
    TILELANG_FA_SOURCE_REPOSITORY="$(git -C "${ROOT}" remote get-url "${remote}")"
fi
export TILELANG_FA_SOURCE_COMMIT TILELANG_FA_SOURCE_REPOSITORY
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
manifests = {}
for version in versions:
    export = root / f"tilelang-{version}" / "export"
    manifests[version] = json.loads((export / "manifest.json").read_text())
    artifacts[version] = {
        name: sha(export / name)
        for name in ("partial.cu", "partial.ptx", "partial.sass", "combine.cu", "combine.ptx", "combine.sass")
    }
source_identities = {
    version: {
        "source_repository": manifests[version]["source_repository"],
        "source_commit": manifests[version]["source_commit"],
        "source_sha256": manifests[version]["source_sha256"],
    }
    for version in versions
}
if len({json.dumps(value, sort_keys=True) for value in source_identities.values()}) != 1:
    raise SystemExit(f"source identity mismatch: {source_identities}")
artifact_parity = {
    name: "bit-identical" if artifacts[versions[0]][name] == artifacts[versions[1]][name] else "different"
    for name in artifacts[versions[0]]
}
result = {
    "schema": 1,
    "output_parity": "bit-identical",
    "case_hashes": case_hashes,
    "source_identity": source_identities[versions[0]],
    "artifact_sha256": artifacts,
    "cross_version_artifact_parity": artifact_parity,
}
path = root / "matrix.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(path)
PY
