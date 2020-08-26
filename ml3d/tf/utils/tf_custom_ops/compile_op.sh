#!/bin/bash

# Get TF variables
TF_INC=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# Create a symlink if libtensorflow_framework is not present
unameOut="$(uname -s)"
case "${unameOut}" in
    Linux*)     machine=Linux;;
    Darwin*)    machine=Mac;;
    CYGWIN*)    machine=Cygwin;;
    MINGW*)     machine=MinGw;;
    *)          machine="UNKNOWN:${unameOut}"
esac

if [ ${machine} = "Mac" ]; then

	framework="${TF_LIB}/libtensorflow_framework.dylib"

	if ! [ -f $framework ]; then
		ln "${TF_LIB}/libtensorflow_framework.2.dylib" $framework
	fi

else
	framework="${TF_LIB}/libtensorflow_framework.so"

	if ! [ -f $framework ]; then
		echo "not exists"
		ln "${TF_LIB}/libtensorflow_framework.so.2" $framework
	fi

fi


# Neighbors op
g++ -std=c++11 -shared tf_neighbors/tf_neighbors.cpp tf_neighbors/neighbors/neighbors.cpp cpp_utils/cloud/cloud.cpp -o tf_neighbors.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared tf_neighbors/tf_batch_neighbors.cpp tf_neighbors/neighbors/neighbors.cpp cpp_utils/cloud/cloud.cpp -o tf_batch_neighbors.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0

# Subsampling op
g++ -std=c++11 -shared tf_subsampling/tf_subsampling.cpp tf_subsampling/grid_subsampling/grid_subsampling.cpp cpp_utils/cloud/cloud.cpp -o tf_subsampling.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
g++ -std=c++11 -shared tf_subsampling/tf_batch_subsampling.cpp tf_subsampling/grid_subsampling/grid_subsampling.cpp cpp_utils/cloud/cloud.cpp -o tf_batch_subsampling.so -fPIC -I$TF_INC -I$TF_INC/external/nsync/public -L$TF_LIB -ltensorflow_framework -O2 -D_GLIBCXX_USE_CXX11_ABI=0
