#!/bin/bash
  
if [[ "$#" -ne 1 ]]; then
    echo "Please, provide the base directory to store the dataset."
    exit 1
fi

if ! command -v unzip &> /dev/null
then
    echo "Error: unzip could not be found. Please, install it to continue"
    exit
fi

BASE_DIR="$1"/ShapeNet

mkdir -p ${BASE_DIR}

export url="https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_segmentation_benchmark_v0.zip"

wget -c -N -O ${BASE_DIR}'/shapenetcore_partanno_segmentation_benchmark_v0.zip' ${url} --no-check-certificate

cd ${BASE_DIR}

unzip shapenetcore_partanno_segmentation_benchmark_v0.zip

mkdir -p ${BASE_DIR}/zip_files
mv shapenetcore_partanno_segmentation_benchmark_v0.zip ${BASE_DIR}/zip_files
