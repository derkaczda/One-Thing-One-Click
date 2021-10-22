FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN apt install -y git python3 python3-dev python3-pip

RUN apt install -y libsparsehash-dev libboost-all-dev
RUN apt install -y libjpeg-dev zlib1g-dev
RUN pip3 install plyfile==0.7.4
RUN pip3 install tensorboardX
RUN pip3 install pyyaml==5.4.1
RUN pip3 install scipy==1.7.1
RUN pip3 install cmake==3.21.3
RUN pip3 install pillow==8.4.0
RUN pip3 install six==1.16.0
RUN pip3 install tqdm==4.62.3
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
RUN ln -s /otoc/lib/spconv/"$(find build -type d -name "lib.*" -print)"/spconv/libspconv.so /otoc/3D-U-Net/spconv/libspconv.so

# Compile pointgroup_ops
# need to change the architecture to match
# gpu
WORKDIR /otoc/lib/pointgroup_ops
RUN python3 setup.py develop

WORKDIR /otoc/3D-U-Net

