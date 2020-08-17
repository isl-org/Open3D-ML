export BASE_DIR="/Users/sanskara/data/Paris_Lille3D"

mkdir -p $BASE_DIR

url_train = "https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2F&files=training_10_classes"
url_test = "https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2F&files=test_10_classes"

wget -c -N url_train -P $BASE_DIR
wget -c -N url_test -P $BASE_DIR

cd $BASE_DIR

tar -xvzf training_10_classes.tar

tar -xvzf test_10_classes.tar
