FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PATH="/root/anaconda3/bin:$PATH"
# \TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"

# RUN apt-get update --allow-releaseinfo-change && apt-get install -y libgbm-dev -y \
#     software-properties-common dirmngr -y \
#     build-essential -y \
#     libgl1-mesa-glx libxrender1 libfontconfig1 -y \
#     libglib2.0-0 -y \
#     libsm6 libxext6 libxrender-dev -y \
#     vim zip unzip wget screen -y \
#     gnupg2 -y \
#     libgl1-mesa-glx -y \
#     git libmagickwand-dev -y

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
        python3-pip \
        python3-dev \
        python3-opencv

# Install Packages
COPY requirements.txt /
RUN python3 -m pip install --upgrade pip
RUN pip3 install -r requirements.txt


# Copy all files in the current directory to the working directory
WORKDIR /app
COPY . /app


CMD [ "/bin/bash" ]