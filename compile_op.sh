#!/bin/bash
cd utils/nearest_neighbors
python setup.py install --home="."
cd ../../

cd cpp_wrappers
sh compile_wrappers.sh
cd ../../../
