#!/usr/bin/env bash
#
#
# 1. clone Open3D-ML repo
#
#
# 2. clone and build Open3D
# For now we have to clone the o3dml_integration branch of open3d
# git clone --recursive --branch o3dml_integration  https://github.com/intel-isl/Open3D.git
#
# install deps + tf + torch
# 
# build open3d only with relevant modules enabled to minimize build time
#       -DBUNDLE_OPEN3D_ML=ON \
#       -DOPEN3D_ML_ROOT=path_to_open3d_ml_repo_root \
#       -DBUILD_TENSORFLOW_OPS=ON \
#       -DBUILD_PYTORCH_OPS=ON \
#       -DBUILD_RPC_INTERFACE=OFF
#       -DBUILD_UNIT_TESTS=OFF \
#       -DBUILD_BENCHMARKS=OFF \
#       -DBUILD_EXAMPLES=OFF \
# 
# 3. install wheel
#
# 4. run examples/tests in the Open3D-ML repo (these do not exist yet)
#
