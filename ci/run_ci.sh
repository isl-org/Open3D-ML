#!/usr/bin/env bash
#
# The following environment variables are required:
# - NPROC
#
TENSORFLOW_VER="2.8.2"
TORCH_GLNX_VER="1.13.1+cpu"
# OPENVINO_DEV_VER="2021.4.2"  # Numpy version conflict with TF 2.8.2
PIP_VER="21.1.1"
WHEEL_VER="0.38.4"
STOOLS_VER="67.3.2"
YAPF_VER="0.30.0"
PYTEST_VER="7.1.2"
PYTEST_RANDOMLY_VER="3.8.0"

set -euo pipefail

echo 1. Prepare the Open3D-ML repo and install dependencies
echo
export PATH_TO_OPEN3D_ML=$(pwd)
# the build system of the main repo expects a master branch. make sure master exists
git checkout -b master || true
python -m pip install -U pip==$PIP_VER \
    wheel=="$WHEEL_VER" \
    setuptools=="$STOOLS_VER" \
    yapf=="$YAPF_VER" \
    pytest=="$PYTEST_VER" \
    pytest-randomly=="$PYTEST_RANDOMLY_VER"

python -m pip install -r requirements.txt
echo $PATH_TO_OPEN3D_ML
cd ..
python -m pip install -U Cython

echo 2. clone Open3D and install dependencies
echo
git clone --recursive --branch master https://github.com/isl-org/Open3D.git

./Open3D/util/install_deps_ubuntu.sh assume-yes
python -m pip install -U tensorflow-cpu==$TENSORFLOW_VER \
    torch==${TORCH_GLNX_VER} --extra-index-url https://download.pytorch.org/whl/cpu/
# openvino-dev=="$OPENVINO_DEV_VER"

echo 3. Configure for bundling the Open3D-ML part
echo
mkdir Open3D/build
pushd Open3D/build
cmake -DBUNDLE_OPEN3D_ML=ON \
    -DOPEN3D_ML_ROOT=$PATH_TO_OPEN3D_ML \
    -DGLIBCXX_USE_CXX11_ABI=OFF \
    -DBUILD_TENSORFLOW_OPS=ON \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_GUI=ON \
    -DBUILD_UNIT_TESTS=OFF \
    -DBUILD_BENCHMARKS=OFF \
    -DBUILD_EXAMPLES=OFF \
    ..

echo 4. Build and install wheel
echo
make -j"$NPROC" install-pip-package

echo 5. run examples/tests in the Open3D-ML repo outside of the repo directory to
echo make sure that the installed package works.
echo
popd
mkdir test_workdir
pushd test_workdir
mv $PATH_TO_OPEN3D_ML/tests .
echo Add --randomly-seed=SEED to the test command to reproduce test order.
python -m pytest tests

echo ... now do the same but in dev mode by setting OPEN3D_ML_ROOT
export OPEN3D_ML_ROOT=$PATH_TO_OPEN3D_ML
echo Add --randomly-seed=SEED to the test command to reproduce test order.
python -m pytest tests
unset OPEN3D_ML_ROOT

popd
