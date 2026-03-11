# FROM osrf/ros:noetic-desktop-full-focal
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai

# 设置时区
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime && \
    apt-get update && \
    apt-get install -y tzdata && \
    dpkg-reconfigure -f noninteractive tzdata

# 安装额外依赖
# RUN apt-get update && apt-get install -y \
#     wget \
#     git \
#     curl \
#     jq \
#     libgl1 \
#     libglib2.0-0 \
#     libx11-6 \
#     libsm6 \
#     ffmpeg \
#     ca-certificates \
#     && rm -rf /var/lib/apt/lists/*

RUN apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-rosbag \
    python3-roslib \
    python3-std-msgs \
    python3-sensor-msgs \
    wget \
    git \
    curl \
    jq \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libsm6 \
    ffmpeg \
    ca-certificates \
    unzip \
    bc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN python3 -m venv /opt/venv --system-site-packages
ENV PATH=/opt/venv/bin:$PATH
RUN mkdir -p /app/lerobot
# Keep only the minimal installable subset to reduce image size.
COPY lerobot/ /app/lerobot/

RUN pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install -e /app/lerobot --no-deps -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN pip install \
    'torch==2.7.1+cpu' \
    'torchvision==0.22.1+cpu' \
    'datasets>=2.19.0,<=3.6.0' \
    huggingface-hub \
    accelerate \
    hydra-core \
    av \
    jsonlines \
    Pillow \
    joblib \
    h5py \
    scikit-learn \
    oss2 \
    opencv-python-headless \
    requests \
    tqdm \
    packaging \
    einops \
    psutil \
    termcolor \
    drake==1.19.0 \
    'pyarrow>=10.0.0' \
    'pyyaml>=6.0' \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install 'numpy==1.26.4' --force-reinstall --no-deps \
    -i https://pypi.tuna.tsinghua.edu.cn/simple

RUN curl -o ossutil-2.1.2-linux-amd64.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.2/ossutil-2.1.2-linux-amd64.zip && \
    unzip ossutil-2.1.2-linux-amd64.zip && \
    cd ossutil-2.1.2-linux-amd64 && \
    chmod 755 ossutil && \
    mv ossutil /usr/local/bin/ && \
    ln -s /usr/local/bin/ossutil /usr/bin/ossutil && \
    cd .. && \
    rm -rf ossutil-2.1.2-linux-amd64 ossutil-2.1.2-linux-amd64.zip

# 复制应用文件
COPY kuavo_data /app/kuavo_data

# 设置脚本可执行权限
RUN chmod +x /app/kuavo_data/*.sh

CMD ["/bin/bash"]