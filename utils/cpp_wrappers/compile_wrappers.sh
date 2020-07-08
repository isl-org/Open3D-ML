#!/bin/bash

# Compile cpp subsampling
cd cpp_subsampling
python3 setup.py build_ext --inplace
cd ..

