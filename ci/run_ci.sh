#!/usr/bin/env bash
#
# The following environment variables are required:
# - NPROC
#
TENSORFLOW_VER="2.5.0"
TORCH_GLNX_VER="1.8.1+cpu"
YAPF_VER="0.30.0"
PYTEST_VER="6.0.1"
PYTEST_RANDOMLY_VER="3.8.0"

set -euo pipefail

echo 1. Prepare the Open3D-ML repo and install dependencies
echo
export PATH_TO_OPEN3D_ML=$(pwd)
# the build system of the main repo expects a master branch. make sure master exists
git checkout -b master || true
pip install -r requirements.txt
echo $PATH_TO_OPEN3D_ML
cd ..
python -m pip install -U Cython

echo 2. clone Open3D and install dependencies
echo
git clone --recursive --branch master https://github.com/isl-org/Open3D.git

./Open3D/util/install_deps_ubuntu.sh assume-yes
python -m pip install -U tensorflow-cpu==$TENSORFLOW_VER
python -m pip install -U torch==${TORCH_GLNX_VER} -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install -U pytest=="$PYTEST_VER" \
    pytest-randomly=="$PYTEST_RANDOMLY_VER"
python -m pip install -U yapf=="$YAPF_VER"

echo 3. Configure for bundling the Open3D-ML part
echo
mkdir Open3D/build
pushd Open3D/build
cmake -DBUNDLE_OPEN3D_ML=ON \
    -DOPEN3D_ML_ROOT=$PATH_TO_OPEN3D_ML \
    -DBUILD_TENSORFLOW_OPS=ON \
    -DBUILD_PYTORCH_OPS=ON \
    -DBUILD_GUI=OFF \
    -DBUILD_RPC_INTERFACE=OFF \
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
echo Add --rondomly-seed=SEED to the test command to reproduce test order.
python -m pytest tests

echo ... now do the same but in dev mode by setting OPEN3D_ML_ROOT
export OPEN3D_ML_ROOT=$PATH_TO_OPEN3D_ML
echo Add --rondomly-seed=SEED to the test command to reproduce test order.
python -m pytest tests
unset OPEN3D_ML_ROOT

popd
