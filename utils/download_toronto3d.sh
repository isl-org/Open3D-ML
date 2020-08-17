export BASE_DIR="/Users/sanskara/data/"

mkdir -p $BASE_DIR

url = "https://1drv.ms/u/s!Amlc6yZnF87psX6hKS8VOQllVvj4?e=yWhrYX"

wget -c -N url -P $BASE_DIR

cd $BASE_DIR

tar -xvzf Toronto_3D.zip

rm Toronto_3D.zip
