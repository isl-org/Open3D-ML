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

BASE_DIR="$1"/Paris_Lille3D

mkdir -p $BASE_DIR

export url_train="https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2F&files=training_10_classes"
export url_test="https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2F&files=test_10_classes"

wget -c -N -O $BASE_DIR'/training_10_classes.zip'  $url_train
wget -c -N -O $BASE_DIR'/test_10_classes.zip' $url_test

cd $BASE_DIR

unzip test_10_classes.zip
unzip training_10_classes.zip

mkdir -p $BASE_DIR/zip_files
mv test_10_classes.zip $BASE_DIR/zip_files
mv training_10_classes.zip $BASE_DIR/zip_files
