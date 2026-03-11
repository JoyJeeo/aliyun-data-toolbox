FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Shanghai

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    curl \
    unzip \
    ffmpeg \
    linux-libc-dev \
    python3-dev \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime \
    && rm -rf /var/lib/apt/lists/*

# Optional OSS utility for upload/download workflows.
RUN curl -fsSLo /tmp/ossutil.zip https://gosspublic.alicdn.com/ossutil/v2/2.1.2/ossutil-2.1.2-linux-amd64.zip \
    && unzip /tmp/ossutil.zip -d /tmp \
    && mv /tmp/ossutil-2.1.2-linux-amd64/ossutil /usr/local/bin/ossutil \
    && chmod 755 /usr/local/bin/ossutil \
    && ln -sf /usr/local/bin/ossutil /usr/local/bin/ossutil64 \
    && rm -rf /tmp/ossutil.zip /tmp/ossutil-2.1.2-linux-amd64

RUN pip install --upgrade pip setuptools wheel \
    && pip install \
      numpy==1.26.4 \
      pandas==2.2.3 \
      pyarrow==23.0.1 \
      jsonlines==4.0.0 \
      packaging==25.0 \
      huggingface-hub==0.35.3 \
      datasets==4.1.1 \
      deepdiff==8.6.0 \
    && pip install \
      lerobot==0.4.4

COPY run.sh /app/run.sh
COPY merge_data.py /app/merge_data.py

RUN chmod +x /app/run.sh

CMD ["/bin/bash"]
