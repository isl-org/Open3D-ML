#!/bin/bash
  
if [ "$#" -ne 1 ]; then
    echo "Please, provide the base directory to store the dataset."
    exit 1
fi

BASE_DIR="$1"/Lyft

mkdir -p $BASE_DIR

url_train="https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/3d-object-detection/train.tar"
url_test="https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/3d-object-detection/test.tar"

wget -c -N -O $BASE_DIR'/train.tar' $url_train
wget -c -N -O $BASE_DIR'/test.tar' $url_test

cd $BASE_DIR

tar -xvf train.tar
tar -xvf test.tar

mkdir tar_files
mv train.tar tar_files
mv test.tar tar_files
