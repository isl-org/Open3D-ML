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

BASE_DIR="$1"/sunrgbd

mkdir -p ${BASE_DIR}

url_sunrgbd="http://rgbd.cs.princeton.edu/data/SUNRGBD.zip"
url_2dbb="http://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat"
url_3dbb="http://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat"
url_toolbox="http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip"

wget -c -N -O ${BASE_DIR}'/SUNRGBD.zip' ${url_sunrgbd} --no-check-certificate
wget -c -N -O ${BASE_DIR}'/SUNRGBDMeta2DBB_v2.mat' ${url_2dbb} --no-check-certificate
wget -c -N -O ${BASE_DIR}'/SUNRGBDMeta3DBB_v2.mat' ${url_3dbb} --no-check-certificate
wget -c -N -O ${BASE_DIR}'/SUNRGBDtoolbox.zip' ${url_toolbox} --no-check-certificate

cd ${BASE_DIR}

unzip SUNRGBD.zip
unzip SUNRGBDtoolbox.zip

mkdir -p ${BASE_DIR}/zip_files
mv SUNRGBDtoolbox.zip SUNRGBD.zip ${BASE_DIR}/zip_files
