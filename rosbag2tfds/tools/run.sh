#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# SETUP
OUTPUT_DIR="temp"

##############################################
# 组装成功标签
##############################################
SUCCESS_LABELS="tfds_success"

if [[ -n "${SUCCESS_ADDITIONAL_LABELS:-}" ]]; then
  # 1) 把 "逗号前后出现的空白" 统一删掉，再去掉头尾空白
  _trimmed=$(echo "${SUCCESS_ADDITIONAL_LABELS}" |
    sed -E 's/[[:space:]]*,[[:space:]]*/,/g' |
    sed -E 's/^[[:space:]]+|[[:space:]]+$//g')

  # 2) 过滤掉因输入形如 ",foo,," 产生的空标签
  IFS=',' read -r -a _parts <<<"${_trimmed}"
  _cleaned=()
  for p in "${_parts[@]}"; do
    [[ -n "$p" ]] && _cleaned+=("$p") # 忽略空串
  done

  # 3) 追加到 LABELS
  if ((${#_cleaned[@]})); then
    SUCCESS_LABELS+=","$(
      IFS=','
      echo "${_cleaned[*]}"
    )
  fi
fi

########################################
# defer-like 收尾：统一失败处理
########################################
cleanup() {
  local status=$? # 捕获最后一次命令的退出码
  set +e          # 关闭 -e，避免 cocli 失败递归触发

  if [[ "${status}" -ne 0 ]]; then
    echo "⚠️  脚本异常退出（exit code=${status}），给记录打失败标签..."
    cocli record update "$COS_RECORDID" --append-labels tfds_failed || true
  fi

  # 无论成功或失败都清理临时目录
  rm -rf "$OUTPUT_DIR"
}
trap cleanup EXIT
trap 'echo "❌ 发生错误，行号: $LINENO";' ERR # 可选：行号提示

##################################
# 初始化 cocli 客户端（如果环境变量存在）
if [[ -n "${COS_PROJECTID:-}" && -n "${COS_TOKEN:-}" ]]; then
  echo "========== 初始化 cocli 客户端 =========="
  if cocli login set -p "$COS_PROJECTID" -t "$COS_TOKEN"; then
    echo "✅ cocli 初始化成功"
  else
    echo "⚠️ cocli 初始化失败，但继续执行（某些功能可能不可用）"
  fi
  echo ""
fi

##################################
# Step 0: 判断当前记录有无执行过转换
echo "========== Step 0: 获取 记录 数据 =========="
echo "当前记录ID: $COS_RECORDID"
echo "项目ID: $COS_PROJECTID"

# 检查是否已成功转换
HAS_RLDS_SUCCESS=$(cocli record describe "$COS_RECORDID" -o json | jq -r '.labels[]?.display_name' | grep -w tfds_success || true)
if [[ -n "$HAS_RLDS_SUCCESS" ]]; then
  echo "✅ 当前记录已存在 tfds_success 标签，跳过转换"
  echo "💡 如需重新转换，请先删除 tfds_success 标签"
  echo "💡 转换结果保存在: /cos/outputs/"
  exit 0
fi

# Step 1: 获取当前记录的 metadata.json 文件
echo "========== Step 1: 获取 Metadata 数据 =========="
echo "当前记录ID: $COS_RECORDID"
echo "文件存储路径: $COS_FILE_VOLUME"
echo "项目ID: $COS_PROJECTID"
echo ""

# metadata.json 保存路径
METADATA_JSON="/tmp/metadata.json"
METADATA_RAW_JSON="/tmp/metadata_raw.json"

# Step 1.1: 从 CoScene API 获取 data_id
echo "正在从 CoScene 记录获取 data_id..."

# 准备 Basic Auth（如果未设置）
if [ -z "${basicAuth:-}" ]; then
  if [ -n "${COS_TOKEN:-}" ]; then
    # COS_TOKEN 应该是 "apiKey:apiSecret" 格式，需要 base64 编码
    basicAuth=$(echo -n "$COS_TOKEN" | base64)
    echo "   使用 COS_TOKEN 生成 Basic Auth"
  else
    echo "⚠️  未设置 basicAuth 或 COS_TOKEN 环境变量"
    echo "   无法调用 CoScene API 获取 data_id"
  fi
fi

# 从 CoScene API 获取 data_id
if [ -z "${data_id:-}" ] && [ -n "${basicAuth:-}" ]; then
  set +e  # 临时关闭错误退出
  data_id=$(curl --location --request GET "https://openapi.coscene.cn/dataplatform/v1alpha1/projects/$COS_PROJECTID/records/$COS_RECORDID" \
    --header "Authorization: Basic ${basicAuth}" \
    --header 'Accept: */*' \
    --header 'Host: openapi.coscene.cn' \
    --header 'Connection: keep-alive' \
    2>/dev/null | jq -r '.customFieldValues[] | select(.property.name=="data_id") | .text.value // empty' 2>/dev/null)
  curl_exit_code=$?
  set -e  # 重新启用错误退出

  if [ $curl_exit_code -eq 0 ] && [ -n "$data_id" ]; then
    echo "✅ 从 CoScene API 获取到 data_id: $data_id"
  else
    echo "⚠️  无法从 CoScene API 获取 data_id (curl exit code: $curl_exit_code)"
    data_id=""
  fi
fi

# Step 1.2: 使用 data_id 从 Kuavo API 获取完整的 metadata
if [ -n "${data_id:-}" ] && [ -n "${kuavoAuth:-}" ]; then
  echo "正在从 Kuavo API 获取完整 metadata (data_id=$data_id)..."

  set +e  # 临时关闭错误退出
  if curl -sS -L -H "kuavo-auth: bearer $kuavoAuth" \
    "http://gym.lejurobot.com/api/kuavo-task/data/detail-with-mark?id=$data_id" \
    | jq '.data' > "$METADATA_RAW_JSON" 2>/dev/null; then

    set -e  # 重新启用错误退出
    echo "✅ metadata_raw.json 下载成功"
  else
    set -e  # 重新启用错误退出
    echo "❌ metadata_raw.json 下载失败"
    rm -f "$METADATA_RAW_JSON"
  fi
else
  if [ -z "${data_id:-}" ]; then
    echo "⚠️  未找到 data_id，跳过 Kuavo API 调用"
  fi
  if [ -z "${kuavoAuth:-}" ]; then
    echo "⚠️  未设置 kuavoAuth，跳过 Kuavo API 调用"
  fi
fi
echo ""

# Step 1.3: 处理 metadata_raw.json 生成最终的 metadata.json
if [ -f "$METADATA_RAW_JSON" ] && [ -s "$METADATA_RAW_JSON" ]; then
  echo "正在处理 metadata_raw.json 生成 metadata.json..."

  if jq '{
    primaryScene: (.topScene // ""),
    primarySceneCode: (.topSceneCode // ""),
    deviceSn: (.deviceSn // ""),
    secondaryScene: (.scene // ""),
    secondarySceneCode: (.sceneCode // ""),
    initSceneText: (.initSceneText // ""),
    tertiarySceneCode: (.subSceneCode // ""),
    taskGroupCode: (.taskGroupCode // ""),
    taskCode: (.taskCode // ""),
    tertiaryScene: (.subScene // ""),
    taskGroupName: (.taskGroupName // ""),
    englishInitSceneText: (.englishInitSceneText // ""),
    taskName: (.taskName // ""),
    globalInstruction: (
      (.marks // []) |
      map(.enSkillDetail // "") |
      map(select(. != "")) |
      join(", ")
    ),
    eefType: (.eefType // ""),
    marks: (
      (.marks // []) | map({
        markStart: (.markStart // ""),
        taskID: (.taskID // ""),
        markEnd: (.markEnd // ""),
        duration: (.duration // ""),
        skillDetail: (.skillDetail // ""),
        startPosition: ((.startPosition|tostring) // ""),
        markType: (.markType // ""),
        endPosition: ((.endPosition|tostring) // ""),
        skillAtomic: (.skillAtomic // ""),
        enSkillDetail: (.enSkillDetail // "")
      })
    )
  }' "$METADATA_RAW_JSON" > "$METADATA_JSON" 2>/dev/null; then
    echo "✅ metadata.json 生成成功"
    rm -f "$METADATA_RAW_JSON"
  else
    echo "❌ metadata.json 生成失败"
    rm -f "$METADATA_RAW_JSON"
    echo '{}' > "$METADATA_JSON"
  fi
else
  # 如果没有从 Kuavo API 获取到数据，检查是否有已存在的 metadata.json
  if [ -f "$COS_FILE_VOLUME/metadata.json" ]; then
    echo "⚠️  未能从 Kuavo API 获取数据，使用已存在的 metadata.json"
    cp "$COS_FILE_VOLUME/metadata.json" "$METADATA_JSON"
  else
    echo "⚠️  未能获取 metadata，创建空 metadata.json"
    echo '{}' > "$METADATA_JSON"
  fi
fi

# 显示最终的 metadata.json
echo "📊 最终 metadata.json 内容:"
jq '.' "$METADATA_JSON" 2>/dev/null || echo "  (无法解析 JSON)"
echo ""

# Step 2: 执行 ROSbag 到 Open-X RLDS 转换
echo "========== Step 2: ROSbag 转换处理 =========="
echo "转换脚本: tools/rosbag_to_tfds.py"
echo "输入目录: /cos/files"
echo "输出目录: /cos/outputs"
echo ""

# 查找 rosbag 文件（在 /cos/files 下）
COS_FILES_DIR="/cos/files"
ROSBAG_FILES=$(find "${COS_FILES_DIR}" -name "*.bag" 2>/dev/null)
ROSBAG_COUNT=$(echo "${ROSBAG_FILES}" | grep -v "^$" | wc -l)
if [[ $ROSBAG_COUNT -eq 0 ]]; then
  echo "❌ 未在 /cos/files 下找到 .bag 文件"
  exit 1
fi

echo "📊 发现 $ROSBAG_COUNT 个 rosbag 文件"

# rosbag_to_tfds.py 期望的目录结构:
#   input_root/
#     subdir1/file1.bag + file1.json
#     subdir2/file2.bag + file2.json
# 需要创建这种结构

INPUT_ROOT="/tmp/tfds_input"
rm -rf "$INPUT_ROOT"
mkdir -p "$INPUT_ROOT"

echo "🔧 准备输入目录结构..."
for BAG_FILE in $ROSBAG_FILES; do
  BAG_BASENAME=$(basename "$BAG_FILE" .bag)
  BAG_PARENT_DIR=$(dirname "$BAG_FILE")

  # 为每个 bag 创建子目录
  SUBDIR="$INPUT_ROOT/$BAG_BASENAME"
  mkdir -p "$SUBDIR"

  # 创建软链接到 bag 文件
  ln -sf "$BAG_FILE" "$SUBDIR/"
  echo "  ✅ 链接 bag: $BAG_FILE -> $SUBDIR/"

  # 查找并链接同名的 json 文件
  JSON_FILE="$BAG_PARENT_DIR/${BAG_BASENAME}.json"
  if [[ -f "$JSON_FILE" ]]; then
    ln -sf "$JSON_FILE" "$SUBDIR/"
    echo "  ✅ 链接 json: $JSON_FILE"
  elif [[ -f "$METADATA_JSON" ]]; then
    # 使用 metadata.json 作为 sidecar
    ln -sf "$METADATA_JSON" "$SUBDIR/${BAG_BASENAME}.json"
    echo "  ✅ 链接 metadata: $METADATA_JSON -> ${BAG_BASENAME}.json"
  fi
done

echo ""
echo "开始执行 ROSbag 转换..."
START_TIME=$(date +%s)

# 创建输出目录
OUTPUT_ROOT="/cos/outputs"
mkdir -p "$OUTPUT_ROOT"

# 使用数组构建参数（更安全的方法）
# 检测脚本运行位置，自动调整路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == "/app" ]]; then
  # 在镜像中运行，使用 /app/tools/
  ROSBAG_TO_OPENX_SCRIPT="/app/tools/rosbag_to_tfds.py"
else
  # 在挂载模式下运行，使用相对路径
  ROSBAG_TO_OPENX_SCRIPT="tools/rosbag_to_tfds.py"
fi

# rosbag_to_tfds.py 期望:
#   --input_root: 包含子目录的根目录，每个子目录有 .bag 和 .json
#   --output_dir: 输出 TFDS 的目录
ARGS=(
  "python3" "$ROSBAG_TO_OPENX_SCRIPT"
  "--input_root" "$INPUT_ROOT"
  "--output_dir" "$OUTPUT_ROOT"
)

# sidecar metadata 已在上面创建目录结构时处理
# 检查是否有 metadata，决定是否启用 --clip_to_marks
if [[ -f "$METADATA_JSON" ]]; then
  ARGS+=("--clip_to_marks")
  echo "✅ 启用 --clip_to_marks 参数"
fi

# 添加 URDF 路径（如果存在，会在 /cos/files 下查找）
if [[ -n "${URDF_PATH:-}" ]]; then
  ARGS+=("--urdf" "$URDF_PATH")
  echo "✅ 设置 URDF 路径: $URDF_PATH"
elif [[ -f "$COS_FILES_DIR/biped_s49/urdf/biped_s49.urdf" ]]; then
  ARGS+=("--urdf" "$COS_FILES_DIR/biped_s49/urdf/biped_s49.urdf")
  echo "✅ 使用默认 URDF 路径: $COS_FILES_DIR/biped_s49/urdf/biped_s49.urdf"
else
  # 尝试在 /cos/files 下查找 URDF
  URDF_FILE=$(find "$COS_FILES_DIR" -name "*.urdf" -type f 2>/dev/null | head -n1)
  if [[ -n "$URDF_FILE" ]]; then
    ARGS+=("--urdf" "$URDF_FILE")
    echo "✅ 找到 URDF 文件: $URDF_FILE"
  else
    echo "⚠️ 未找到 URDF 文件，TCP 位姿将使用 TF 或设置为零"
  fi
fi

# 添加转换相关的可选参数（使用环境变量或默认值）
# timeline
TIMELINE="${timeline:-camera}"
ARGS+=("--timeline" "$TIMELINE")
echo "✅ 设置 --timeline $TIMELINE"

# eef_type
EEF_TYPE="${eef_type:-auto}"
ARGS+=("--eef" "$EEF_TYPE")
echo "✅ 设置 --eef $EEF_TYPE"

# split
SPLIT="${split:-train}"
ARGS+=("--split" "$SPLIT")
echo "✅ 设置 --split $SPLIT"

# dataset_name (仅用于后续合并，不传给 rosbag_to_tfds.py)
DATASET_NAME="${dataset_name:-delivery_openx}"
echo "📝 数据集名称: $DATASET_NAME (用于后续合并)"

# 相机相关参数（使用环境变量或默认值）
# main_rgb_topic
MAIN_RGB_TOPIC="${main_rgb_topic:-/cam_h/color/image_raw/compressed}"
ARGS+=("--main_rgb_topic" "$MAIN_RGB_TOPIC")
echo "✅ 设置 --main_rgb_topic $MAIN_RGB_TOPIC"

# rgb_topics
RGB_TOPICS="${rgb_topics:-/cam_h/color/image_raw/compressed,/cam_l/color/image_raw/compressed,/cam_r/color/image_raw/compressed}"
ARGS+=("--rgb_topics" "$RGB_TOPICS")
echo "✅ 设置 --rgb_topics $RGB_TOPICS"

# depth_topics
DEPTH_TOPICS="${depth_topics:-/cam_h/depth/image_raw/compressed,/cam_l/depth/image_rect_raw/compressed,/cam_r/depth/image_rect_raw/compressed}"
ARGS+=("--depth_topics" "$DEPTH_TOPICS")
echo "✅ 设置 --depth_topics $DEPTH_TOPICS"

# camera_info_topics
CAMERA_INFO_TOPICS="${camera_info_topics:-/cam_h/color/camera_info,/cam_l/color/camera_info,/cam_r/color/camera_info}"
ARGS+=("--camera_info_topics" "$CAMERA_INFO_TOPICS")
echo "✅ 设置 --camera_info_topics $CAMERA_INFO_TOPICS"

# camera_link_hints (支持 JSON 或逗号分隔格式)
CAMERA_LINK_HINTS="${camera_link_hints:-head:camera,left:l_hand_camera,right:r_hand_camera}"
ARGS+=("--camera_link_hints" "$CAMERA_LINK_HINTS")
echo "✅ 设置 --camera_link_hints $CAMERA_LINK_HINTS"

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

# 显示最终执行的命令（用于调试）
echo "📝 执行命令: ${ARGS[*]}"
echo ""

# 执行命令
if timeout 36000 "${ARGS[@]}"; then

  END_TIME=$(date +%s)
  DURATION=$((END_TIME - START_TIME))
  echo ""
  echo "✅ ROSbag 转换成功完成！"
  echo "⏱️  转换耗时: ${DURATION} 秒"

  # 新脚本输出到 OUTPUT_ROOT/<bag_basename>/
  FINAL_OUTPUT_DIR="$OUTPUT_ROOT/$BAG_BASENAME"
  
  # 显示输出文件统计
  if [[ -d "$FINAL_OUTPUT_DIR" ]]; then
    OUTPUT_SIZE=$(du -sh "$FINAL_OUTPUT_DIR" | cut -f1)
    OUTPUT_FILES=$(find "$FINAL_OUTPUT_DIR" -type f | wc -l)
    echo "📊 输出文件大小: $OUTPUT_SIZE"
    echo "📊 输出文件数量: $OUTPUT_FILES 个"
    echo "📂 输出目录: $FINAL_OUTPUT_DIR"
  else
    echo "⚠️ 输出目录不存在: $FINAL_OUTPUT_DIR"
    FINAL_OUTPUT_DIR="$OUTPUT_ROOT"
  fi

else
  echo "❌ ROSbag 转换失败"
  exit 1
fi
echo ""

# Step 3: 显示转换结果信息
echo "========== Step 3: 转换结果 =========="
echo "✅ 转换完成，结果已保存在: $FINAL_OUTPUT_DIR"
echo "📂 输出目录: $FINAL_OUTPUT_DIR"
echo "💡 文件直接保留在 /cos/outputs 目录下，后续可由同事修改"

# Step 4: 给当前记录打 tfds_success 标签
echo "========== Step 4: 打标签 =========="
echo "为当前记录 $COS_RECORDID 添加标签 tfds_success ..."
if cocli record update "$COS_RECORDID" --append-labels tfds_success; then
  echo "✅ 已成功添加标签 tfds_success"
else
  echo "❌ 添加标签失败"
  exit 1
fi

# Step 5: 合并所有 episodes 到最终 TFDS 格式
echo "========== Step 5: 合并 Episodes 到 TFDS 格式 =========="

# 检测脚本运行位置，自动调整路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == "/app" ]]; then
  # 在镜像中运行，使用 /app/tools/
  COMBINE_SCRIPT="/app/tools/combine_episodes_to_tfds.py"
else
  # 在挂载模式下运行，使用相对路径
  COMBINE_SCRIPT="tools/combine_episodes_to_tfds.py"
fi

# 设置合并参数
OUTPUT_ROOT="/cos/outputs"
FINAL_MERGED_DIR="$OUTPUT_ROOT/delivery_openx_final"
DATASET_NAME="${dataset_name:-delivery_openx}"
SPLIT="${split:-train}"

echo "合并脚本: $COMBINE_SCRIPT"
echo "输入目录: $OUTPUT_ROOT"
echo "输出目录: $FINAL_MERGED_DIR"
echo ""

echo "开始合并 episodes..."
COMBINE_START_TIME=$(date +%s)

COMBINE_ARGS=(
  "python3" "$COMBINE_SCRIPT"
  "--output_dir" "$OUTPUT_ROOT"
  "--merged_dir" "$FINAL_MERGED_DIR"
  "--dataset_name" "$DATASET_NAME"
  "--split" "$SPLIT"
)

echo "📝 执行命令: ${COMBINE_ARGS[*]}"
echo ""

if timeout 36000 "${COMBINE_ARGS[@]}"; then
  COMBINE_END_TIME=$(date +%s)
  COMBINE_DURATION=$((COMBINE_END_TIME - COMBINE_START_TIME))
  echo ""
  echo "✅ Episodes 合并成功！"
  echo "⏱️  合并耗时: ${COMBINE_DURATION} 秒"

  # 检查最终输出目录
  FINAL_TFDS_DIR="$FINAL_MERGED_DIR/$DATASET_NAME/1.0.0"
  if [[ -d "$FINAL_TFDS_DIR" ]]; then
    echo "📂 最终 TFDS 目录: $FINAL_TFDS_DIR"
    echo "📊 目录结构:"
    ls -lh "$FINAL_TFDS_DIR" || true

    # 显示 train 目录内容
    TRAIN_DIR="$FINAL_TFDS_DIR/$SPLIT"
    if [[ -d "$TRAIN_DIR" ]]; then
      echo ""
      echo "📊 $SPLIT 目录内容:"
      ls -lh "$TRAIN_DIR" | head -20 || true

      # 统计 tfrecord 文件
      TFRECORD_COUNT=$(find "$TRAIN_DIR" -name "*.tfrecord*" -type f 2>/dev/null | wc -l)
      TOTAL_SIZE=$(du -sh "$TRAIN_DIR" 2>/dev/null | cut -f1)
      echo ""
      echo "📦 TFRecord 文件数: $TFRECORD_COUNT"
      echo "📊 总大小: $TOTAL_SIZE"
    fi
  else
    echo "⚠️ 最终 TFDS 目录不存在: $FINAL_TFDS_DIR"
    echo "⚠️ 将使用原始转换结果上传"
    FINAL_TFDS_DIR=""
  fi
else
  echo "❌ Episodes 合并失败"
  echo "⚠️ 将使用原始转换结果上传"
  FINAL_TFDS_DIR=""
fi
echo ""

# Step 6: 上传最终数据集到指定记录
echo "========== Step 6: 上传到指定记录 =========="

if [[ -z "${recordName:-}" ]]; then
  echo "⚠️  未设置 recordName 环境变量，跳过上传"
  if [[ -n "$FINAL_TFDS_DIR" && -d "$FINAL_TFDS_DIR" ]]; then
    echo "💡 最终 TFDS 数据集已保存在: $FINAL_TFDS_DIR"
  else
    echo "💡 转换结果已保存在: $FINAL_OUTPUT_DIR"
  fi
  echo "💡 如需上传到 CoScene，请设置 recordName 环境变量"
else
  echo "目标记录ID: $recordName"

  # 优先上传合并后的 TFDS 数据集，如果不存在则上传原始转换结果
  UPLOAD_DIR=""
  if [[ -n "$FINAL_TFDS_DIR" && -d "$FINAL_TFDS_DIR" ]]; then
    UPLOAD_DIR="$FINAL_TFDS_DIR"
    echo "上传类型: 合并后的 TFDS 数据集"
    echo "上传目录: $UPLOAD_DIR"
  elif [[ -d "$FINAL_OUTPUT_DIR" ]]; then
    UPLOAD_DIR="$FINAL_OUTPUT_DIR"
    echo "上传类型: 原始转换结果"
    echo "上传目录: $UPLOAD_DIR"
  else
    echo "❌ 没有可上传的目录"
    exit 1
  fi

  # 复制到 /cos/files 目录（cocli 只能访问当前记录的文件）
  TEMP_UPLOAD_DIR="$COS_FILE_VOLUME/rlds_upload_temp"
  rm -rf "$TEMP_UPLOAD_DIR"
  mkdir -p "$TEMP_UPLOAD_DIR"

  echo "正在复制文件到 $TEMP_UPLOAD_DIR ..."
  if cp -r "$UPLOAD_DIR" "$TEMP_UPLOAD_DIR/"; then
    echo "✅ 文件复制成功"
  else
    echo "❌ 文件复制失败"
    rm -rf "$TEMP_UPLOAD_DIR"
    exit 1
  fi

  # 列出临时目录内容（调试用）
  echo "临时目录内容:"
  ls -lhR "$TEMP_UPLOAD_DIR/" | head -50 || ls -lh "$TEMP_UPLOAD_DIR/"

  # 上传文件到目标记录（使用相对路径）
  echo "正在上传文件到记录 $recordName ..."
  if cocli record file copy "$COS_RECORDID" "$recordName" --files "rlds_upload_temp/" -f; then
    echo "✅ 已成功上传文件到 $recordName"

    if [[ -n "$FINAL_TFDS_DIR" && -d "$FINAL_TFDS_DIR" ]]; then
      echo "📂 上传的 TFDS 数据集结构:"
      echo "   rlds_upload_temp/"
      echo "   └── 1.0.0/"
      echo "       ├── dataset_info.json"
      echo "       ├── features.json"
      echo "       └── $SPLIT/"
      echo "           ├── $DATASET_NAME-$SPLIT.tfrecord-00000-of-XXXXX"
      echo "           └── ..."
    fi
  else
    echo "❌ 上传失败"
    rm -rf "$TEMP_UPLOAD_DIR"
    exit 1
  fi

  # 清理临时目录
  rm -rf "$TEMP_UPLOAD_DIR"
  echo "✅ 临时文件清理完成"
fi

echo "🎉 所有处理完成！"

exit 0

