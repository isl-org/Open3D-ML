#!/usr/bin/env bash

set -euo pipefail

PIP_VER="24.3.1"
OPEN3D_REPO="isl-org/Open3D"
RELEASE_TAG="main-devel"
PY_TAG="cp312"

echo "1. Download the latest open3d_cpu devel wheel from ${OPEN3D_REPO}@${RELEASE_TAG}"
echo
gh release download "${RELEASE_TAG}" \
    --repo "${OPEN3D_REPO}" \
    --pattern "open3d_cpu-*-${PY_TAG}-${PY_TAG}-manylinux*_x86_64.whl" \
    --dir . \
    --clobber
WHEEL_PATH="$(ls open3d_cpu-*-"${PY_TAG}"-*.whl)"
echo "Downloaded: ${WHEEL_PATH}"

echo "2. Install the wheel in a fresh virtual environment"
echo
python -m venv open3d_test.venv
# shellcheck disable=SC1091
source open3d_test.venv/bin/activate
python -m pip install -U pip=="${PIP_VER}"
python -m pip install "${WHEEL_PATH}"

echo "3. Sanity-check the installed package"
echo
python -W default -c "
import open3d
print('Installed:', open3d)
print('BUILD_PYTORCH_OPS:', open3d._build_config['BUILD_PYTORCH_OPS'])
print('BUILD_SYCL_MODULE:', open3d._build_config['BUILD_SYCL_MODULE'])
"

echo "4. Install Open3D-ML's own dependencies (torch flavor + base requirements)"
echo
export PATH_TO_OPEN3D_ML="$PWD"
python -m pip install -r requirements.txt -r requirements-torch.txt \
    -r requirements-tensorflow.txt

echo "5. Run the Open3D-ML pytest suite against the installed wheel"
echo
echo "Add --randomly-seed=SEED to the test command to reproduce test order."
./tests/run_tests.sh

echo "6. Also verify the OPEN3D_ML_ROOT dev-mode path (bundled models loaded from the"
echo "   Open3D-ML checkout instead of whatever the wheel itself bundled)"
echo
export OPEN3D_ML_ROOT="${PATH_TO_OPEN3D_ML}"
./tests/run_tests.sh
unset OPEN3D_ML_ROOT

deactivate
