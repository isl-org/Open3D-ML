#!/bin/bash
cd ml3d/ops/cpp_wrappers
sh compile_wrappers.sh
cd ..

cd tf_custom_ops
sh compile_op.sh
cd ../../../../
