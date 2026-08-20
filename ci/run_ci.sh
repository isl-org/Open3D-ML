#!/usr/bin/env bash
#
# Install a prebuilt Open3D wheel + Open3D-ML deps, then run tests.
#
# Usage: ./ci/run_ci.sh [cpu|cuda|xpu]
#
#   cpu  — open3d_cpu wheel + requirements-torch.txt
#   cuda — default open3d manylinux wheel + requirements-torch-cuda.txt
#   xpu  — open3d_xpu wheel + requirements-torch-xpu.txt
#
# Pytest skips (parity, accelerator-only ops, missing ops) are handled in the
# test modules. See docs/howtos.md.

set -euo pipefail

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \?//'
    echo
    echo "Example: ./ci/run_ci.sh cpu"
    exit "${1:-0}"
}

BACKEND="${1:-cpu}"
case "${BACKEND}" in
    cpu | cuda | xpu) ;;
    -h | --help) usage 0 ;;
    *)
        echo "Unknown backend: ${BACKEND}" >&2
        usage 1
        ;;
esac

PIP_VER="24.3.1"
OPEN3D_REPO="isl-org/Open3D"
RELEASE_TAG="main-devel"
PY_TAG="cp312"

case "${BACKEND}" in
    cpu)
        WHEEL_GLOB="open3d_cpu-*-${PY_TAG}-${PY_TAG}-manylinux*_x86_64.whl"
        TORCH_REQUIREMENTS="requirements-torch.txt"
        ;;
    cuda)
        WHEEL_GLOB="open3d-[0-9]*-${PY_TAG}-${PY_TAG}-manylinux*_x86_64.whl"
        TORCH_REQUIREMENTS="requirements-torch-cuda.txt"
        ;;
    xpu)
        WHEEL_GLOB="open3d_xpu-*-${PY_TAG}-${PY_TAG}-manylinux*_x86_64.whl"
        TORCH_REQUIREMENTS="requirements-torch-xpu.txt"
        ;;
esac

echo "Open3D-ML CI backend: ${BACKEND}"
echo "  wheel pattern: ${WHEEL_GLOB}"
echo "  torch requirements: ${TORCH_REQUIREMENTS}"
echo

echo "1. Download the latest Open3D devel wheel from ${OPEN3D_REPO}@${RELEASE_TAG}"
echo
gh release download "${RELEASE_TAG}" \
    --repo "${OPEN3D_REPO}" \
    --pattern "${WHEEL_GLOB}" \
    --dir . \
    --clobber
shopt -s nullglob
wheel_matches=( ${WHEEL_GLOB} )
shopt -u nullglob
if [ "${#wheel_matches[@]}" -ne 1 ]; then
    echo "Expected exactly one wheel matching ${WHEEL_GLOB}, found ${#wheel_matches[@]}" >&2
    exit 1
fi
WHEEL_PATH="${wheel_matches[0]}"
echo "Downloaded: ${WHEEL_PATH}"

echo "2. Install the wheel in a fresh virtual environment"
echo
python -m venv open3d_test.venv
# shellcheck disable=SC1091
source open3d_test.venv/bin/activate
python -m pip install -U pip=="${PIP_VER}" pytest
python -m pip install "${WHEEL_PATH}"

echo "3. Sanity-check the installed package"
echo
python -W default -c "import open3d; print('Installed:', open3d)"

echo "4. Install Open3D-ML dependencies"
echo
export PATH_TO_OPEN3D_ML="$PWD"
python -m pip install -r requirements.txt -r "${TORCH_REQUIREMENTS}" \
    -r requirements-tensorflow.txt

run_test_suite() {
    echo "Running: ./tests/run_tests.sh tests"
    echo "Add --randomly-seed=SEED via pytest env if reproducing order."
    ./tests/run_tests.sh tests
}

echo "5. Run the Open3D-ML pytest suite against the installed wheel"
echo
run_test_suite

echo "6. Also verify the OPEN3D_ML_ROOT dev-mode path (models loaded from this"
echo "   checkout instead of the wheel bundle)"
echo
export OPEN3D_ML_ROOT="${PATH_TO_OPEN3D_ML}"
run_test_suite
unset OPEN3D_ML_ROOT

deactivate
