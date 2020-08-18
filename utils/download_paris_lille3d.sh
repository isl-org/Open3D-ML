export BASE_DIR="/home/sanskar_agrawal/data/test/Paris_Lille3D"

mkdir -p $BASE_DIR

export url_train="https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2F&files=training_10_classes"
export url_test="https://cloud.mines-paristech.fr/index.php/s/JhIxgyt0ALgRZ1O/download?path=%2F&files=test_10_classes"

wget -c -N -O $BASE_DIR'/training_10_classes.zip'  $url_train
wget -c -N -O $BASE_DIR'/test_10_classes.zip' $url_test

cd $BASE_DIR

unzip test_10_classes.zip
unzip train_10_classes.zip

rm test_10_classes.zip
rm train_10_classes.zip

