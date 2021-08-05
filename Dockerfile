FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu18.04
SHELL ["/bin/bash", "-c"]
RUN nvcc --version

# Core system packages
RUN apt-get update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make

# Install new version of CMake
RUN apt purge --auto-remove cmake
RUN wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc 2>/dev/null | gpg --dearmor - | tee /etc/apt/trusted.gpg.d/kitware.gpg >/dev/null
RUN apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
RUN lsb_release -a
RUN apt update
RUN apt install -y cmake

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Additional dev packages
RUN apt update
RUN apt install -y libssl-dev libmodule-install-perl libboost-all-dev libgl1-mesa-dev libopenblas-dev

# Install torch prerequisites
RUN conda install python=3.8
RUN conda install pip 
RUN pip install https://github.com/isl-org/open3d_downloads/releases/download/torch1.7.1/torch-1.7.1-cp38-cp38-linux_x86_64.whl
RUN conda install open3d -c open3d-admin -c conda-forge
RUN conda install tensorboard matplotlib
RUN python -c "import torch; print(torch.__version__)"
RUN python -c "import open3d as o3d"

ENV OPEN3D_ML_ROOT /Open3D-ML
WORKDIR /Open3D-ML