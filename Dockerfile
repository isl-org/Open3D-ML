FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
RUN nvcc --version

# Set the timezone info because otherwise tzinfo blocks install 
# flow and ignores the non-interactive frontend command 🤬🤬🤬
RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime && echo "/usr/share/zoneinfo/America/New_York" > /etc/timezone

# Core system packages
RUN apt-get update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Additional dev packages
RUN apt install -y --no-install-recommends libssl-dev libmodule-install-perl libboost-all-dev libgl1-mesa-dev libopenblas-dev

# Install torch prerequisites
RUN conda install python=3.8
RUN conda install pip 
RUN conda install cudatoolkit=11.1 -c conda-forge
RUN pip install open3d
RUN python -c "import open3d as o3d"
RUN pip install https://github.com/isl-org/open3d_downloads/releases/download/torch1.7.1/torch-1.7.1-cp38-cp38-linux_x86_64.whl
RUN python -c "import torch; print(torch.__version__)"
RUN conda install tensorboard matplotlib

ENV OPEN3D_ML_ROOT /Open3D-ML
WORKDIR /Open3D-ML