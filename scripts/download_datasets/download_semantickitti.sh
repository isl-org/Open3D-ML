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

BASE_DIR="$1"/SemanticKitti

mkdir -p $BASE_DIR

url_data="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
url_label="http://semantic-kitti.org/assets/data_odometry_labels.zip"

wget -c -N -O $BASE_DIR'/data_odometry_velodyne.zip' $url_data
wget -c -N -O $BASE_DIR'/data_odometry_labels.zip' $url_label

cd $BASE_DIR

unzip data_odometry_velodyne.zip
unzip data_odometry_labels.zip

mkdir zip_files
mv data_odometry_labels.zip zip_files
mv data_odometry_velodyne.zip zip_files
