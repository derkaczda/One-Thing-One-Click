FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6" \
    CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    CUDA_ARCHITECTURES=8.6 \
    CUDA_LAUNCH_BLOCKING=1

RUN apt update && apt install -y \
    git=1:2.25.1-1ubuntu3.2 \
    python3=3.8.2-0ubuntu2 \
    python3-dev=3.8.2-0ubuntu2 \
    python3-pip=20.0.2-5ubuntu1.6 \
    libsparsehash-dev=2.0.3-1 \
    libboost-all-dev=1.71.0.0ubuntu2 \
    libjpeg-dev=8c-2ubuntu8 \
    zlib1g-dev=1:1.2.11.dfsg-2ubuntu1.2 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install \
    plyfile==0.7.4 \
    tensorboardX \
    pyyaml==5.4.1 \
    scipy==1.7.1 \
    cmake==3.21.3 \
    pillow==8.4.0 \
    six==1.16.0 \
    tqdm==4.62.3 \
    matplotlib
# torch 1.9.1 throws an illegal memory access error
# in the batchnorm module (but only sometimes)
RUN pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

RUN git clone --recursive https://github.com/derkaczda/One-Thing-One-Click /otoc

# Install spconv
# we need to copy the build binary into the different
# subprojects or we get an error
RUN cd /otoc/lib/spconv && \
    python3 setup.py bdist_wheel && \
    pip3 install dist/*.whl && \
    cp "$(find /otoc/lib/spconv/build -type d -name "lib.*" -print)"/spconv/libspconv.so /otoc/3D-U-Net/spconv/libspconv.so && \
    cp "$(find /otoc/lib/spconv/build -type d -name "lib.*" -print)"/spconv/libspconv.so /otoc/relation/spconv/libspconv.so && \
    cp "$(find /otoc/lib/spconv/build -type d -name "lib.*" -print)"/spconv/libspconv.so /otoc/merge/spconv/libspconv.so && \
    rm -rf /otoc/lib/spconv/build

# Compile pointgroup_ops
# need to change the architecture to match gpu
RUN cd /otoc/lib/pointgroup_ops && python3 setup.py develop
ENV PYTHONPATH=/otoc:$PYTHONPATH
