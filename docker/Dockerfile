# 使用具有 CUDA 11.0 的基础镜像
FROM swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# 安装基本工具和依赖
RUN sed -i s:/archive.ubuntu.com:/mirrors.tuna.tsinghua.edu.cn/ubuntu:g /etc/apt/sources.list
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# 输出版本信息
RUN python3 --version
RUN nvcc --version
RUN cmake --version

# 安装yolo包
RUN python3 -m pip install --upgrade pip
RUN pip3 install ultralytics


# 将工作目录设置为 /workspace
WORKDIR /workspace

