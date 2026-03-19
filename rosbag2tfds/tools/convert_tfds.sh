#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ============================================================
# convert_tfds.sh - 批量转换 ROSbag 到 TFDS
# 
# 期望的输入目录结构（由 fetch_metadata.sh 生成）:
#   /cos/files/
#     ├── episode_001/
#     │   ├── episode_001.bag
#     │   └── episode_001.json
#     ├── episode_002/
#     │   ├── episode_002.bag
#     │   └── episode_002.json
#     └── ...
#
# 输出:
#   /cos/outputs/delivery_openx/1.0.0/
#     ├── dataset_info.json
#     ├── features.json
#     └── train/
#         ├── delivery_openx-train.tfrecord-00000-of-XXXXX
#         └── ...
# ============================================================

echo "========== Convert TFDS Script =========="
echo "输入目录: ${COS_FILE_VOLUME:-/cos/files}"
echo "输出目录: /cos/outputs"
echo "当前记录ID: ${COS_RECORDID:-未设置}"
echo "项目ID: ${COS_PROJECTID:-未设置}"
echo ""

COS_FILE_VOLUME="${COS_FILE_VOLUME:-/cos/files}"
OUTPUT_ROOT="/cos/outputs"

# 初始化 cocli（用于后续打标签等操作）
if [[ -n "${COS_PROJECTID:-}" && -n "${COS_TOKEN:-}" ]]; then
  echo ">>> 初始化 cocli..."
  cocli login set -p "$COS_PROJECTID" -t "$COS_TOKEN" || echo "⚠️ cocli 初始化失败"
  echo ""
fi

# 检测脚本位置
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == "/app" || "$SCRIPT_DIR" == "/app/tools" ]]; then
  ROSBAG_TO_TFDS_SCRIPT="/app/tools/rosbag_to_tfds.py"
else
  ROSBAG_TO_TFDS_SCRIPT="tools/rosbag_to_tfds.py"
fi

# 检查输入目录结构（排除隐藏目录如 .cos）
echo ">>> 检查输入目录结构..."
SUBDIRS=$(find "$COS_FILE_VOLUME" -mindepth 1 -maxdepth 1 -type d ! -name '.*' 2>/dev/null | wc -l)
if [[ $SUBDIRS -eq 0 ]]; then
  echo "❌ 未在 $COS_FILE_VOLUME 下找到子目录"
  echo "💡 请确保目录结构正确: 每个子目录包含一个 .bag 和 .json 文件"
  exit 1
fi
echo "📊 发现 $SUBDIRS 个子目录（已排除隐藏目录）"

# 检查每个子目录是否包含 .bag 文件（排除隐藏目录）
BAG_COUNT=0
for DIR in $(find "$COS_FILE_VOLUME" -mindepth 1 -maxdepth 1 -type d ! -name '.*'); do
  if ls "$DIR"/*.bag >/dev/null 2>&1; then
    BAG_COUNT=$((BAG_COUNT + 1))
  fi
done
echo "📊 包含 .bag 文件的目录: $BAG_COUNT 个"

if [[ $BAG_COUNT -eq 0 ]]; then
  echo "❌ 没有找到包含 .bag 文件的目录"
  exit 1
fi
echo ""

# 创建输出目录
mkdir -p "$OUTPUT_ROOT"

# 构建参数
ARGS=(
  "python3" "$ROSBAG_TO_TFDS_SCRIPT"
  "--input_root" "$COS_FILE_VOLUME"
  "--output_dir" "$OUTPUT_ROOT"
)

# 可选参数
TIMELINE="${timeline:-camera}"
ARGS+=("--timeline" "$TIMELINE")

EEF_TYPE="${eef_type:-auto}"
ARGS+=("--eef" "$EEF_TYPE")

SPLIT="${split:-train}"
ARGS+=("--split" "$SPLIT")

# 相机参数
MAIN_RGB_TOPIC="${main_rgb_topic:-/cam_h/color/image_raw/compressed}"
ARGS+=("--main_rgb_topic" "$MAIN_RGB_TOPIC")

RGB_TOPICS="${rgb_topics:-/cam_h/color/image_raw/compressed,/cam_l/color/image_raw/compressed,/cam_r/color/image_raw/compressed}"
ARGS+=("--rgb_topics" "$RGB_TOPICS")

DEPTH_TOPICS="${depth_topics:-/cam_h/depth/image_raw/compressed,/cam_l/depth/image_rect_raw/compressed,/cam_r/depth/image_rect_raw/compressed}"
ARGS+=("--depth_topics" "$DEPTH_TOPICS")

CAMERA_INFO_TOPICS="${camera_info_topics:-/cam_h/color/camera_info,/cam_l/color/camera_info,/cam_r/color/camera_info}"
ARGS+=("--camera_info_topics" "$CAMERA_INFO_TOPICS")

# camera_link_hints (支持 JSON 或逗号分隔格式)
# 默认使用 /tf_static 中的相机 frame 名称 (会自动链接 TF 变换链)
# 例如: base_link → zhead_1_link → zhead_2_link → camera
CAMERA_LINK_HINTS="${camera_link_hints:-head:camera,left:l_hand_camera,right:r_hand_camera}"
ARGS+=("--camera_link_hints" "$CAMERA_LINK_HINTS")

# TF topics (支持动态 /tf 和静态 /tf_static)
TF_TOPIC="${tf_topic:-/tf}"
ARGS+=("--tf_topic" "$TF_TOPIC")

TF_STATIC_TOPIC="${tf_static_topic:-/tf_static}"
ARGS+=("--tf_static_topic" "$TF_STATIC_TOPIC")

# TCP frame 参数（如果环境变量存在则传递）
if [[ -n "${TCP_FRAME_LEFT:-}" ]]; then
  ARGS+=("--tcp_frame_left" "$TCP_FRAME_LEFT")
  echo "✅ 设置 --tcp_frame_left $TCP_FRAME_LEFT"
fi

if [[ -n "${TCP_FRAME_RIGHT:-}" ]]; then
  ARGS+=("--tcp_frame_right" "$TCP_FRAME_RIGHT")
  echo "✅ 设置 --tcp_frame_right $TCP_FRAME_RIGHT"
fi

# 启用 clip_to_marks（如果有 json 文件）
ARGS+=("--clip_to_marks")

# URDF 路径
if [[ -n "${URDF_PATH:-}" ]]; then
  ARGS+=("--urdf" "$URDF_PATH")
elif [[ -f "$COS_FILE_VOLUME/biped_s49/urdf/biped_s49.urdf" ]]; then
  ARGS+=("--urdf" "$COS_FILE_VOLUME/biped_s49/urdf/biped_s49.urdf")
else
  URDF_FILE=$(find "$COS_FILE_VOLUME" -name "*.urdf" -type f 2>/dev/null | head -n1)
  if [[ -n "$URDF_FILE" ]]; then
    ARGS+=("--urdf" "$URDF_FILE")
    echo "✅ 找到 URDF: $URDF_FILE"
  fi
fi

echo ">>> 开始转换..."
echo "📝 命令: ${ARGS[*]}"
echo ""

START_TIME=$(date +%s)

if timeout 36000 "${ARGS[@]}"; then
  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  echo ""
  echo "✅ 转换成功！"
  echo "⏱️  耗时: ${DURATION} 秒"
  
  # 显示输出
  TFDS_DIR="$OUTPUT_ROOT/delivery_openx/1.0.0"
  if [[ -d "$TFDS_DIR" ]]; then
    echo ""
    echo "📂 输出目录: $TFDS_DIR"
    ls -lh "$TFDS_DIR/"
    
    TRAIN_DIR="$TFDS_DIR/$SPLIT"
    if [[ -d "$TRAIN_DIR" ]]; then
      TFRECORD_COUNT=$(find "$TRAIN_DIR" -name "*.tfrecord*" -type f 2>/dev/null | wc -l)
      TOTAL_SIZE=$(du -sh "$TRAIN_DIR" 2>/dev/null | cut -f1)
      echo ""
      echo "📦 TFRecord 文件数: $TFRECORD_COUNT"
      echo "📊 总大小: $TOTAL_SIZE"
    fi
  fi
else
  echo "❌ 转换失败"
  exit 1
fi

echo ""
echo "🎉 批量转换完成！"
exit 0

