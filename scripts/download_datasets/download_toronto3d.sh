#!/bin/bash
  
if [ "$#" -ne 1 ]; then
    echo "Please, provide the base directory to store the dataset."
    exit 1
fi

if ! command -v unzip &> /dev/null
then
    echo "Error: unzip could not be found. Please, install it to continue."
    exit
fi

BASE_DIR="$1"/Toronto3D

export url="https://xx9lca.sn.files.1drv.com/y4mUm9-LiY3vULTW79zlB3xp0wzCPASzteId4wdUZYpzWiw6Jp4IFoIs6ADjLREEk1-IYH8KRGdwFZJrPlIebwytHBYVIidsCwkHhW39aQkh3Vh0OWWMAcLVxYwMTjXwDxHl-CDVDau420OG4iMiTzlsK_RTC_ypo3z-Adf-h0gp2O8j5bOq-2TZd9FD1jPLrkf3759rB-BWDGFskF3AsiB3g"

mkdir -p $BASE_DIR

wget -c -N -O $BASE_DIR'/Toronto_3D.zip' $url

cd $BASE_DIR

unzip -j Toronto_3D.zip

# cleanup
mkdir -p $BASE_DIR/zip_files
mv Toronto_3D.zip $BASE_DIR/zip_files

