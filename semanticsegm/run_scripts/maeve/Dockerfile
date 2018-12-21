#arguments
ARG cudaversion=10.0
ARG cudnnversion=7.4.1.5

#get cuda base image
FROM nvidia/cuda:${cudaversion}-devel-ubuntu18.04
#FROM nvcr.io/nvidia/tensorrt:18.11-py3

#Install prereqs
RUN apt-get update && apt-get install --assume-yes apt-utils dialog && apt-get install --assume-yes cpio && \
    apt-get install --assume-yes wget && apt-get install --assume-yes bzip2 && \ 
    apt-get install --assume-yes cmake && apt-get install --assume-yes less && \
    apt-get install --assume-yes lsb-release && apt-get install --assume-yes g++ && \
    apt-get install --assume-yes emacs && apt-get install --assume-yes zip && apt-get install --assume-yes unzip && \
    apt-get install --assume-yes openjdk-8-jdk pkg-config zlib1g-dev git

#Install cudnn:
ARG cudaversion=10.0
ARG cudnnversion=7.4.1.5
COPY cudnn-${cudaversion}-linux-x64-v${cudnnversion}.tgz /
RUN tar -xzf cudnn-${cudaversion}-linux-x64-v${cudnnversion}.tgz && mv cuda/lib64/* /usr/local/cuda/lib64/ && \
    mv cuda/include/* /usr/local/cuda/include/ && mv cuda/NVIDIA_SLA_cuDNN_Support.txt /usr/local/cuda/ && \
    rm cudnn-${cudaversion}-linux-x64-v${cudnnversion}.tgz

#Install NCCL
ARG ncclversion=2.3.7
COPY nccl_${ncclversion}-1+cuda${cudaversion}_x86_64.txz /
RUN tar -xf nccl_${ncclversion}-1+cuda${cudaversion}_x86_64.txz && \
    mv nccl_${ncclversion}-1+cuda${cudaversion}_x86_64 /usr/local/nccl && \
    rm nccl_${ncclversion}-1+cuda${cudaversion}_x86_64.txz

#Install TensorRT
ARG tensorrtversion=5.0.2.6
COPY TensorRT-${tensorrtversion}.Ubuntu-18.04.1.x86_64-gnu.cuda-${cudaversion}.cudnn${cudnnversion}.tgz /
RUN tar -xzf TensorRT-${tensorrtversion}.Ubuntu-18.04.1.x86_64-gnu.cuda-${cudaversion}.cudnn${cudnnversion}.tgz && \
    mv TensorRT-${tensorrtversion} /usr/local/tensorrt && \
    rm TensorRT-${tensorrtversion}.Ubuntu-18.04.1.x86_64-gnu.cuda-${cudaversion}.cudnn${cudnnversion}.tgz

#Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-5.3.1-Linux-x86_64.sh && bash Anaconda3-5.3.1-Linux-x86_64.sh -b && \
    ln -s /root/anaconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && rm -f Anaconda3-5.3.1-Linux-x86_64.sh

#Install python packages
ENV PATH=/root/anaconda3/bin:${PATH}
RUN conda install numpy pip && pip install keras_preprocessing

#Install bazel
ARG bazelversion=0.19.2
#0.20.0
RUN mkdir -p /src/bazel && cd /src/bazel && \
    wget https://github.com/bazelbuild/bazel/releases/download/${bazelversion}/bazel-${bazelversion}-dist.zip && \
    unzip bazel-${bazelversion}-dist.zip && ./compile.sh && rm bazel-${bazelversion}-dist.zip
ENV PATH=${PATH}:/src/bazel/output

#Install tensorflow
ARG smversion=6.1
ENV PYTHON_BIN_PATH=/root/anaconda3/bin/python
ENV PYTHON_LIB_PATH=/root/anaconda3/lib/python3.7/site-packages
ENV GCC_HOST_COMPILER_PATH=/usr/bin/gcc
ENV CC_OPT_FLAGS="-march=native -O3"
ENV TF_NEED_CUDA=1
ENV TF_CUDA_VERSION=${cudaversion}
ENV CUDA_TOOLKIT_PATH=/usr/local/cuda
ENV TF_CUDA_COMPUTE_CAPABILITIES=${smversion}
ENV TF_CUDNN_VERSION=${cudnnversion}
ENV CUDNN_INSTALL_PATH=/usr/local/cuda
ENV TF_NEED_TENSORRT=1
ENV TF_TENSORRT_VERSION=${tensorrtversion}
ENV TENSORRT_INSTALL_PATH=/usr/local/tensorrt
ENV TF_NCCL_VERSION=${ncclversion}
ENV NCCL_INSTALL_PATH=/usr/local/nccl
ENV TF_NEED_AWS=0
ENV TF_NEED_OPENCL_SYCL=0
ENV TF_NEED_ROCM=0
ENV TF_NEED_GCP=0
ENV TF_NEED_HDFS=0
ENV TF_NEED_JEMALLOC=0
ENV TF_NEED_KAFKA=0
ENV TF_DOWNLOAD_CLANG=0
ENV TF_ENABLE_XLA=0
ENV TF_NEED_GDR=0
ENV TF_NEED_VERBS=0
ENV TF_NEED_MPI=0
ENV TF_SET_ANDROID_WORKSPACE=0
ENV TF_NEED_NGRAPH=0
ENV LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:/usr/local/tensorrt/lib:${LD_LIBRARY_PATH}
RUN cd /src && git clone https://github.com/tensorflow/tensorflow.git && cd tensorflow && \
    ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} && \
    yes "" | ./configure && \
    bazel build --action_env=LD_LIBRARY_PATH=${LD_LIBRARY_PATH} --config=cuda -c opt //tensorflow/tools/pip_package:build_pip_package && \
    mkdir -p /src/wheels && \
    bazel-bin/tensorflow/tools/pip_package/build_pip_package /src/wheels && \
    pip install /src/wheels/tensorflow*.whl

#some environment tweaks
RUN echo "umask 0000" >> /etc/bash.bashrc
