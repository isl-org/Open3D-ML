#!/bin/bash
  
if [ "$#" -ne 1 ]; then
    echo "Please, provide the base directory to store the dataset."
    exit 1
fi

if ! command -v unzip &> /dev/null
then
    echo "Error: unzip could not be found. Please, install it to continue"
    exit
fi

BASE_DIR="$1"/Kitti

mkdir -p $BASE_DIR

url_velodyne="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
url_calib="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip"
url_label="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"

wget -c -N -O $BASE_DIR'/data_object_velodyne.zip' $url_velodyne
wget -c -N -O $BASE_DIR'/data_object_calib.zip' $url_calib
wget -c -N -O $BASE_DIR'/data_object_label_2.zip' $url_label

cd $BASE_DIR

unzip data_object_velodyne.zip
unzip data_object_calib.zip
unzip data_object_label_2.zip

mkdir zip_files
mv data_object_velodyne.zip zip_files
mv data_object_calib.zip zip_files
mv data_object_label_2.zip zip_files
