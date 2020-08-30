export BASE_DIR="/home/sanskar_agrawal/data/test/"
export url="https://xx9lca.sn.files.1drv.com/y4mUm9-LiY3vULTW79zlB3xp0wzCPASzteId4wdUZYpzWiw6Jp4IFoIs6ADjLREEk1-IYH8KRGdwFZJrPlIebwytHBYVIidsCwkHhW39aQkh3Vh0OWWMAcLVxYwMTjXwDxHl-CDVDau420OG4iMiTzlsK_RTC_ypo3z-Adf-h0gp2O8j5bOq-2TZd9FD1jPLrkf3759rB-BWDGFskF3AsiB3g"

mkdir -p $BASE_DIR

wget -c -N -O $BASE_DIR'/Toronto_3D.zip' $url

cd $BASE_DIR

unzip Toronto_3D.zip

rm Toronto_3D.zip
