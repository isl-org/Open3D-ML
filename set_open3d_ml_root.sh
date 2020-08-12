# Sets the env var OPEN3D_ML_ROOT to the directory of this file.
# The open3d package will use this var to integrate ml3d into a common namespace.
export OPEN3D_ML_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
if [[ $0 == $BASH_SOURCE ]]; then 
        echo "source this script to set the OPEN3D_ML_ROOT env var."     
else
        echo "OPEN3D_ML_ROOT is now $OPEN3D_ML_ROOT"
fi
