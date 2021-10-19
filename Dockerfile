FROM nvidia/cuda:11.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install -y git python3 python3-dev python3-pip

# Change to python3.7 as default
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.6 1
# RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 2
# RUN update-alternatives --set python3 /usr/bin/python3.7
# RUN python3 --version

RUN apt install -y libsparsehash-dev libboost-all-dev
RUN apt install -y libjpeg-dev zlib1g-dev
RUN pip3 install plyfile
RUN pip3 install tensorboardX
RUN pip3 install pyyaml==5.4.1
RUN pip3 install scipy
RUN pip3 install cmake
RUN pip3 install pillow
RUN pip3 install tqdm
# RUN pip3 install torch
RUN pip3 install torch==1.9.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
ENV TORCH_CUDA_ARCH_LIST "8.6"

RUN git clone --recursive https://github.com/derkaczda/One-Thing-One-Click /otoc

ENV CUDA_HOME /usr/local/cuda
ENV CUDA_ROOT /usr/local/cuda
# ENV CUDAHOSTCXX /usr/local/cuda-11.1/bin/nvcc
# Install spconv
# ENV CXXFLAGS "-fsanitize=address"
# ENV LDFLAGS "-fsanitize=address"
ENV CUDA_ARCHITECTURES 8.6
WORKDIR /otoc/3D-U-Net/lib/spconv
RUN python3 setup.py bdist_wheel
# ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libasan.so.5
# ENV ASAN_OPTIONS "detect_leaks=0"
RUN pip3 install dist/*.whl
RUN ln -s /otoc/3D-U-Net/lib/spconv/"$(find build -type d -name "lib.*" -print)"/spconv/libspconv.so /otoc/3D-U-Net/spconv/libspconv.so

# Compile pointgroup_ops
# need to change the architecture to match
# gpu

WORKDIR /otoc/3D-U-Net/lib/pointgroup_ops
RUN python3 setup.py develop

ENV CUDA_LAUNCH_BLOCKING 1
RUN pip3 install six
WORKDIR /otoc/3D-U-Net

