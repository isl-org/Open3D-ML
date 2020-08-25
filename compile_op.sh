#!/bin/bash
cd utils/nearest_neighbors
python setup.py install --home="."
cd ../../

cd cpp_wrappers
sh compile_wrappers.sh
cd ../

cd ml3d/tf/utils/tf_custom_ops
sh compile_op.sh
cd ../../../../