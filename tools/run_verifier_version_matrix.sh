#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT="${1:?usage: run_verifier_version_matrix.sh OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
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
BASE_SITE_PACKAGES="$("${PYTHON_BIN}" -c 'import site; print(site.getsitepackages()[0])')"
Z3_LIBRARY_DIR="$("${PYTHON_BIN}" -c 'import pathlib, z3; print(pathlib.Path(z3.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${Z3_LIBRARY_DIR}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
for version in 0.1.8 0.1.9; do
    env_dir="${OUTPUT}/venv-${version}"
    result_dir="${OUTPUT}/tilelang-${version}"
    mkdir -p "${result_dir}"
    "${PYTHON_BIN}" -m venv "${env_dir}"
    env_site_packages="$("${env_dir}/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
    printf '%s\n' "${BASE_SITE_PACKAGES}" >"${env_site_packages}/image-base.pth"
    "${env_dir}/bin/pip" install --disable-pip-version-check --no-deps --force-reinstall "tilelang==${version}"
    "${env_dir}/bin/pip" install --disable-pip-version-check --no-deps -e "${ROOT}"
    if [ "${version}" = 0.1.9 ]; then
        set +e
        "${env_dir}/bin/python" "${ROOT}/tools/run_paged_verify_case.py" \
            --output "${result_dir}/vanilla-cases" >"${result_dir}/vanilla-sm70.log" 2>&1
        vanilla_rc=$?
        set -e
        printf '%s\n' "${vanilla_rc}" >"${result_dir}/vanilla-sm70.exit_code"
        test "${vanilla_rc}" -ne 0
        grep -q 'no suitable user-defined conversion from "__nv_bfloat16" to "__half"' \
            "${result_dir}/vanilla-sm70.log"
        common_header="$(
            "${env_dir}/bin/python" - <<'PY'
import pathlib
import tilelang
print(pathlib.Path(tilelang.__file__).parent / "src/tl_templates/cuda/common.h")
PY
        )"
        "${env_dir}/bin/python" - "${common_header}" <<'PY'
import pathlib
import sys
path = pathlib.Path(sys.argv[1])
text = path.read_text()
old = "return __nv_bfloat162{__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)};"
new = """return __nv_bfloat162{
      __float2bfloat16(__bfloat162float(a.x) * __bfloat162float(b.x) +
                       __bfloat162float(c.x)),
      __float2bfloat16(__bfloat162float(a.y) * __bfloat162float(b.y) +
                       __bfloat162float(c.y))};"""
if text.count(old) != 1:
    raise SystemExit("TileLang 0.1.9 SM70 BF16 fallback site changed")
path.write_text(text.replace(old, new))
PY
        export TILELANG_RUNTIME_PATCH=sm70-bf16-fma-fallback
    else
        unset TILELANG_RUNTIME_PATCH || true
    fi
    "${env_dir}/bin/python" "${ROOT}/tools/run_paged_verify_case.py" --output "${result_dir}/cases"
    "${env_dir}/bin/python" "${ROOT}/tools/export_paged_verify.py" \
        --output "${result_dir}/export" --no-causal --num-pages 1024 --max-blocks 1024
done
"${PYTHON_BIN}" - "${OUTPUT}" <<'PY'
import hashlib
import json
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
versions = ("0.1.8", "0.1.9")
vanilla_log = root / "tilelang-0.1.9" / "vanilla-sm70.log"
vanilla_rc = int((root / "tilelang-0.1.9" / "vanilla-sm70.exit_code").read_text())
vanilla_text = vanilla_log.read_text()
if vanilla_rc == 0 or 'no suitable user-defined conversion from "__nv_bfloat16" to "__half"' not in vanilla_text:
    raise SystemExit("TileLang 0.1.9 vanilla SM70 failure changed unexpectedly")
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
if manifests["0.1.8"]["tilelang_runtime_patch"] is not None:
    raise SystemExit("TileLang 0.1.8 must remain unpatched")
if manifests["0.1.9"]["tilelang_runtime_patch"] != "sm70-bf16-fma-fallback":
    raise SystemExit("TileLang 0.1.9 compatibility patch was not recorded")
result = {
    "schema": 1,
    "output_parity": "bit-identical",
    "version_compatibility": {
        "0.1.8": "vanilla-pass",
        "0.1.9": "vanilla-sm70-header-failure; pass after sm70-bf16-fma-fallback",
        "0.1.9_vanilla_exit_code": vanilla_rc,
        "0.1.9_vanilla_log_sha256": sha(vanilla_log),
        "0.1.9_runtime_patch": manifests["0.1.9"]["tilelang_runtime_patch"],
    },
    "case_hashes": case_hashes,
    "source_identity": source_identities[versions[0]],
    "artifact_sha256": artifacts,
    "cross_version_artifact_parity": artifact_parity,
}
path = root / "matrix.json"
path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(path)
PY
