FROM osrf/ros:noetic-desktop

SHELL ["/bin/bash", "-c"]

# 环境变量：避免交互、允许在系统环境直接 pip 安装
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    TZ=Etc/UTC

# 配置 pip 使用国内镜像源（临时禁用，使用官方源）
# RUN mkdir -p /root/.pip && \
#     echo '[global]' > /root/.pip/pip.conf && \
#     echo 'index-url = https://pypi.tuna.tsinghua.edu.cn/simple' >> /root/.pip/pip.conf && \
#     echo 'trusted-host = pypi.tuna.tsinghua.edu.cn' >> /root/.pip/pip.conf && \
#     echo 'timeout = 120' >> /root/.pip/pip.conf

# 复制仓库，以便在构建阶段运行 env_setup/setup_container.sh
WORKDIR /work
COPY . /work

# 在镜像构建阶段完成依赖安装，后续运行容器无需重复执行
RUN chmod +x /work/setup_container.sh && \
    /work/setup_container.sh && \
    rm -rf /var/lib/apt/lists/*

# 复制 tools 目录到 /app/tools
RUN mkdir -p /app/tools
COPY tools/*.py /app/tools/
# 确保 __init__.py 存在（如果不存在则创建）
RUN touch /app/tools/__init__.py 2>/dev/null || true

# 复制脚本到 /app/tools/
COPY tools/run.sh /app/tools/run.sh
COPY tools/fetch_metadata.sh /app/tools/fetch_metadata.sh
COPY tools/convert_tfds.sh /app/tools/convert_tfds.sh
RUN chmod +x /app/tools/*.sh

# 保留旧的 /app/run.sh 入口（兼容）
RUN ln -sf /app/tools/run.sh /app/run.sh

# 设置 PYTHONPATH，确保可以导入 tools 模块
ENV PYTHONPATH=/app${PYTHONPATH:+:$PYTHONPATH}

CMD ["/bin/bash"]
