FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt update && apt install -y \
    git=1:2.25.1-1ubuntu3.2 \
    python3=3.8.2-0ubuntu2 \
    python3-dev=3.8.2-0ubuntu2 \
    python3-pip=20.0.2-5ubuntu1.6 \
    libsparsehash-dev=2.0.3-1 \
    libboost-all-dev=1.71.0.0ubuntu2 \
    libjpeg-dev=8c-2ubuntu8 \
    zlib1g-dev=1:1.2.11.dfsg-2ubuntu1.2 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install plyfile==0.7.4
RUN pip3 install tensorboardX
RUN pip3 install pyyaml==5.4.1
RUN pip3 install scipy==1.7.1
RUN pip3 install cmake==3.21.3
RUN pip3 install pillow==8.4.0
RUN pip3 install six==1.16.0
RUN pip3 install tqdm==4.62.3
# torch 1.9.1 throws an illegal memory access error
# in the batchnorm module (but only sometimes)
RUN pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

ENV TORCH_CUDA_ARCH_LIST "8.6"
ENV CUDA_HOME /usr/local/cuda
ENV CUDA_ROOT /usr/local/cuda
ENV CUDA_ARCHITECTURES 8.6
ENV CUDA_LAUNCH_BLOCKING 1

RUN git clone --recursive https://github.com/derkaczda/One-Thing-One-Click /otoc

# Install spconv
WORKDIR /otoc/lib/spconv
RUN python3 setup.py bdist_wheel
RUN pip3 install dist/*.whl
# we need to link the build binary into the different
# subprojects or we get an error
RUN ln -s /otoc/lib/spconv/"$(find build -type d -name "lib.*" -print)"/spconv/libspconv.so /otoc/3D-U-Net/spconv/libspconv.so
RUN ln -s /otoc/lib/spconv/"$(find build -type d -name "lib.*" -print)"/spconv/libspconv.so /otoc/relation/spconv/libspconv.so
RUN ln -s /otoc/lib/spconv/"$(find build -type d -name "lib.*" -print)"/spconv/libspconv.so /otoc/merge/spconv/libspconv.so

# Compile pointgroup_ops
# need to change the architecture to match
# gpu
WORKDIR /otoc/lib/pointgroup_ops
RUN python3 setup.py develop

WORKDIR /otoc/3D-U-Net

