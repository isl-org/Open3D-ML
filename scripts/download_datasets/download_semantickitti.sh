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

export url_data="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
export url_label="http://semantic-kitti.org/assets/data_odometry_labels.zip"


more << EOF

SemanticKITTI consists of the velodyne scans and the semantic annotation.
The velodyne download link has to be obtained manually by visiting 

http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip

in your browser. After downloading, unzip the data_odometry_velodyne.zip file
to $BASE_DIR


press any key to continue to download and unzip the semantic labels.
EOF

wget -c -N -O $BASE_DIR'/data_odometry_labels.zip' $url_label

cd $BASE_DIR

unzip data_odometry_labels.zip

mkdir zip_files
mv data_odometry_labels.zip zip_files
