#!/usr/bin/env bash
# Bootstrap ROS1 + TensorFlow 依赖，用于在 Docker 容器中执行 rosbag→Open-X 转换。

set -euo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo "请在容器内以 root 权限运行此脚本（或使用 sudo）。" >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PIP_BREAK_SYSTEM_PACKAGES=1

echo ">>> 已假定容器自带 ROS1 软件源，跳过 APT 源配置。"

echo ">>> 安装基础工具 (若已存在会自动跳过)..."
apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  wget \
  python3-pip \
  python3-venv \
  build-essential \
  jq

echo ">>> 安装 cocli (CoScene CLI)..."
# 下载并安装 cocli
if ! command -v cocli &> /dev/null; then
  curl -fsSL https://download.coscene.cn/cocli/install.sh | sh || {
    echo "⚠️ cocli 安装失败，某些功能可能不可用"
  }
else
  echo "✓ cocli 已安装"
fi

echo ">>> 安装 Drake 库 (用于 Forward Kinematics 计算)..."
# 尝试通过 APT 安装 Drake（推荐方式）
DRAKE_INSTALLED=false

# 添加 Drake APT 仓库（如果尚未添加）
if ! grep -q "drake-packages.csail.mit.edu" /etc/apt/sources.list.d/*.list 2>/dev/null; then
  echo "  添加 Drake APT 仓库..."
  # 尝试使用 apt-key（旧方法）
  if wget -qO- https://drake-packages.csail.mit.edu/drake.pub.gpg | apt-key add - 2>/dev/null; then
    echo "deb [arch=amd64] https://drake-packages.csail.mit.edu/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/drake.list
  else
    # 使用新的 GPG keyring 方法（Ubuntu 22.04+）
    mkdir -p /usr/share/keyrings
    wget -qO- https://drake-packages.csail.mit.edu/drake.pub.gpg | gpg --dearmor -o /usr/share/keyrings/drake.gpg 2>/dev/null || true
    if [[ -f /usr/share/keyrings/drake.gpg ]]; then
      echo "deb [arch=amd64 signed-by=/usr/share/keyrings/drake.gpg] https://drake-packages.csail.mit.edu/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/drake.list
    else
      echo "  Warning: Failed to add Drake GPG key, will try pip install instead"
    fi
  fi
  apt-get update -y || true
fi

# 尝试通过 APT 安装
if apt-get install -y --no-install-recommends drake 2>/dev/null; then
  DRAKE_INSTALLED=true
  echo "  ✓ Drake installed via APT"
else
  echo "  Warning: Drake APT package installation failed, trying pip install..."
  # 尝试通过 pip 安装 pydrake
  if python3 -m pip install --no-cache-dir pydrake 2>/dev/null; then
    DRAKE_INSTALLED=true
    echo "  ✓ Drake (pydrake) installed via pip"
  else
    echo "  ⚠ Warning: Both APT and pip installation failed. FK calculator will not be available."
    echo "             TCP pose fields will be set to zeros during conversion."
    echo "             To enable FK, please install Drake manually:"
    echo "             - APT: apt-get install drake"
    echo "             - pip: pip install pydrake"
  fi
fi

echo ">>> 检查 ROS1 (Noetic) 环境..."
if [[ -f "/opt/ros/noetic/setup.bash" ]]; then
  set +u
  source /opt/ros/noetic/setup.bash
  set -u
  rosversion -d
else
  echo "未检测到 /opt/ros/noetic，若容器不是 ROS1 官方镜像，请自行安装 ROS1 再重试。" >&2
  exit 2
fi

echo ">>> 升级 pip 并修复 numpy/scipy 版本..."
# 尝试多个镜像源，如果失败则回退
PIP_MIRRORS=(
  "-i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn"
  "-i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com"
  "-i https://pypi.douban.com/simple/ --trusted-host pypi.douban.com"
)

PIP_MIRROR=""
for mirror in "${PIP_MIRRORS[@]}"; do
  echo "  尝试使用镜像源: $mirror"
  if python3 -m pip install --upgrade pip $mirror 2>&1 | tee /tmp/pip_upgrade.log; then
    PIP_MIRROR="$mirror"
    echo "  ✓ 镜像源可用"
    break
  else
    echo "  ⚠ 镜像源失败，尝试下一个..."
    sleep 1
  fi
done

# 如果所有镜像源都失败，使用官方源
if [[ -z "$PIP_MIRROR" ]]; then
  echo "  ⚠ 所有镜像源都失败，使用官方 PyPI 源..."
  PIP_MIRROR=""
  python3 -m pip install --upgrade pip || {
    echo "  ❌ pip 升级失败，但继续执行..."
  }
fi

python3 -m pip install --no-cache-dir $PIP_MIRROR "numpy<2" "scipy<1.11"

echo ">>> 安装 TensorFlow / TFDS / RLDS 等 Python 依赖..."
# 使用已选定的镜像源安装
if [[ -n "$PIP_MIRROR" ]]; then
  echo "  使用镜像源: $PIP_MIRROR"
else
  echo "  使用官方 PyPI 源"
fi

# 尝试使用镜像源安装，失败则回退到官方源
if ! python3 -m pip install --no-cache-dir --default-timeout=600 $PIP_MIRROR \
  tensorflow==2.13.* \
  tensorflow-datasets>=4.9.0 \
  "apache-beam[gcp]==2.54.0" \
  "dm-reverb[tensorflow]" \
  rlds \
  absl-py \
  pillow \
  opencv-python \
  matplotlib \
  "protobuf<4.22" \
  pyarrow \
  pydot; then
  echo "  ⚠ 镜像源安装失败，回退到官方 PyPI 源..."
  # 临时禁用 pip.conf 中的镜像源配置
  if [[ -f /root/.pip/pip.conf ]]; then
    mv /root/.pip/pip.conf /root/.pip/pip.conf.bak
  fi
  python3 -m pip install --no-cache-dir --default-timeout=600 \
    tensorflow==2.13.* \
    tensorflow-datasets>=4.9.0 \
    "apache-beam[gcp]==2.54.0" \
    "dm-reverb[tensorflow]" \
    rlds \
    opencv-python \
    pillow \
    tqdm \
    matplotlib \
    "protobuf<4.22" \
    pyarrow \
    pydot
  # 恢复 pip.conf
  if [[ -f /root/.pip/pip.conf.bak ]]; then
    mv /root/.pip/pip.conf.bak /root/.pip/pip.conf
  fi
fi

if [[ -f "${REPO_ROOT}/delivery-dataset-demo/requirements.txt" ]]; then
  python3 -m pip install --no-cache-dir $PIP_MIRROR -r "${REPO_ROOT}/delivery-dataset-demo/requirements.txt"
fi

echo ">>> 校验 Python 运行环境..."
python3 - <<'PY' || echo "Warning: Python validation failed, but continuing..."
import rosbag
print(f"rosbag module ok (rosbag={rosbag.__name__})")

# 检查 TensorFlow（但不导入，避免 AVX 错误）
try:
    import importlib.util
    tf_spec = importlib.util.find_spec("tensorflow")
    if tf_spec is not None:
        print("✓ TensorFlow is installed")
    else:
        print("⚠ TensorFlow is NOT installed")
except Exception as e:
    print(f"⚠ TensorFlow check failed: {e}")

# 检查 Drake 是否可用
try:
    from pydrake.multibody.parsing import Parser
    from pydrake.multibody.plant import MultibodyPlant
    print("✓ Drake (pydrake) is available - FK calculator will work")
except ImportError:
    print("⚠ Drake (pydrake) is NOT available - FK calculator will be disabled")
    print("  TCP pose fields will be set to zeros during conversion")
PY

echo ">>> 运行工具自检 (仅打印 --help)..."
python3 "${REPO_ROOT}/tools/rosbag_to_tfds.py" --help >/tmp/rosbag_to_tfds_help.txt 2>&1 || true
head -n 20 /tmp/rosbag_to_tfds_help.txt 2>/dev/null || echo "  (跳过 help 输出)"

echo ">>> 完成。请在容器内执行 rosbag 转换："
echo "    python3 tools/rosbag_to_tfds.py --bag <path/to.bag> --output_root /work/output"
