#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

# ============================================================
# fetch_metadata.sh - 从 Kuavo API 拉取 metadata 并组织目录结构
# 
# 功能：
#   1. 从 CoScene API 获取 data_id
#   2. 使用 data_id 从 Kuavo API 获取完整 metadata
#   3. 找到 bag 文件，创建同名目录
#   4. 将 bag 文件移动到目录中，json 文件命名为同名
#
# 输入目录结构:
#   /cos/files/xxx.bag
#
# 输出目录结构:
#   /cos/files/xxx/xxx.bag + xxx.json
# ============================================================

echo "========== Fetch Metadata Script =========="
echo "当前记录ID: ${COS_RECORDID:-未设置}"
echo "项目ID: ${COS_PROJECTID:-未设置}"
echo "文件目录: ${COS_FILE_VOLUME:-/cos/files}"
echo ""

COS_FILE_VOLUME="${COS_FILE_VOLUME:-/cos/files}"

# 初始化 cocli
if [[ -n "${COS_PROJECTID:-}" && -n "${COS_TOKEN:-}" ]]; then
  echo ">>> 初始化 cocli..."
  cocli login set -p "$COS_PROJECTID" -t "$COS_TOKEN" || echo "⚠️ cocli 初始化失败"
  echo ""
fi

# Step 1: 获取 data_id
echo ">>> Step 1: 获取 data_id..."

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
data_id=""
if [ -n "${basicAuth:-}" ] && [ -n "${COS_PROJECTID:-}" ] && [ -n "${COS_RECORDID:-}" ]; then
  set +e
  data_id=$(curl --location --request GET "https://openapi.coscene.cn/dataplatform/v1alpha1/projects/$COS_PROJECTID/records/$COS_RECORDID" \
    --header "Authorization: Basic ${basicAuth}" \
    --header 'Accept: */*' \
    --header 'Host: openapi.coscene.cn' \
    --header 'Connection: keep-alive' \
    2>/dev/null | jq -r '.customFieldValues[] | select(.property.name=="data_id") | .text.value // empty' 2>/dev/null)
  curl_exit_code=$?
  set -e
  if [ $curl_exit_code -eq 0 ] && [ -n "$data_id" ]; then
    echo "✅ 获取到 data_id: $data_id"
  else
    echo "⚠️ 未能获取 data_id (curl exit code: $curl_exit_code)"
    data_id=""
  fi
fi
echo ""

# Step 2: 获取 metadata
echo ">>> Step 2: 获取 metadata..."
METADATA_JSON="/tmp/metadata.json"
METADATA_VALID=false

if [ -n "${data_id:-}" ] && [ -n "${kuavoAuth:-}" ]; then
  echo "正在从 Kuavo API 获取 metadata..."
  set +e
  METADATA_RAW=$(curl -sS -L -H "kuavo-auth: bearer $kuavoAuth" \
    "https://gym.lejurobot.com/api/kuavo-task/data/detail-with-mark?id=$data_id" 2>/dev/null)
  set -e
  
  if echo "$METADATA_RAW" | jq -e '.data' >/dev/null 2>&1; then
    echo "$METADATA_RAW" | jq '.data | {
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
        map(.enDesc // "") |
        map(select(. != "")) |
        map(gsub("\\.$"; "")) |
        if length == 0 then ""
        elif length == 1 then .[0] | if . == "" then "" else (.[0:1] | ascii_upcase) + .[1:] + "." end
        else
          map(if . == "" then "" else (.[0:1] | ascii_downcase) + .[1:] end) |
          if length == 2 then "First \(.[0]), and then \(.[1])."
          elif length == 3 then "First \(.[0]), then \(.[1]), finally \(.[2])."
          elif length == 4 then "First \(.[0]), then \(.[1]), next \(.[2]), finally \(.[3])."
          else "First \(.[0]), " + (.[1:-1] | map("then \(.)") | join(", ")) + ", finally \(.[-1])."
          end
        end
      ),
      taskRemark: (.taskRemark // ""),
      globalInstructionVariants: (
        (.taskRemark // "") |
        if . == "" then []
        else gsub("[；\\n]"; ";") | split(";") | map(gsub("^\\s+|\\s+$"; "")) | map(select(. != ""))
        end
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
          enDesc: (.enDesc // ""),
          endPosition: ((.endPosition|tostring) // ""),
          skillAtomic: (.skillAtomic // ""),
          enSkillDetail: (.enSkillDetail // "")
        })
      )
    }' > "$METADATA_JSON"
    METADATA_VALID=true
    echo "✅ metadata.json 生成成功"
    echo "📄 metadata 内容预览:"
    jq '.' "$METADATA_JSON" 2>/dev/null | head -20
  else
    echo "❌ metadata 解析失败"
    METADATA_VALID=false
  fi
else
  [ -z "${data_id:-}" ] && echo "⚠️ 未找到 data_id"
  [ -z "${kuavoAuth:-}" ] && echo "⚠️ 未设置 kuavoAuth"
  METADATA_VALID=false
fi
echo ""

# 检查是否成功获取了有效的 metadata
if [[ "$METADATA_VALID" != "true" ]]; then
  echo "❌ 未能获取有效的 metadata，终止处理"
  echo "💡 请检查:"
  echo "   1. 该 record 是否设置了 data_id 字段"
  echo "   2. kuavoAuth 环境变量是否正确"
  echo "   3. Kuavo API 是否可访问"
  exit 1
fi

# Step 3: 组织目录结构（使用临时目录，因为 /cos/files 是只读的）
echo ">>> Step 3: 组织目录结构..."

# 创建临时输出目录
TEMP_OUTPUT_DIR="/tmp/organized_bags"
rm -rf "$TEMP_OUTPUT_DIR"
mkdir -p "$TEMP_OUTPUT_DIR"

ROSBAG_FILES=$(find "${COS_FILE_VOLUME}" -maxdepth 1 -name "*.bag" 2>/dev/null || true)
ROSBAG_COUNT=$(echo "${ROSBAG_FILES}" | grep -v "^$" | wc -l)

if [[ $ROSBAG_COUNT -eq 0 ]]; then
  echo "❌ 未在 $COS_FILE_VOLUME 下找到 .bag 文件"
  exit 1
fi

echo "📊 发现 $ROSBAG_COUNT 个 rosbag 文件"

for BAG_FILE in $ROSBAG_FILES; do
  BAG_BASENAME=$(basename "$BAG_FILE" .bag)
  TARGET_DIR="$TEMP_OUTPUT_DIR/$BAG_BASENAME"

  echo "处理: $BAG_BASENAME"
  mkdir -p "$TARGET_DIR"

  # 创建 bag 文件的软链接（节省空间和时间）
  ln -sf "$BAG_FILE" "$TARGET_DIR/$BAG_BASENAME.bag"
  echo "  ✅ 链接 bag -> $TARGET_DIR/"

  # 复制 metadata.json（已验证有效）
  cp "$METADATA_JSON" "$TARGET_DIR/${BAG_BASENAME}.json"
  echo "  ✅ 创建 json: ${BAG_BASENAME}.json"
done

echo ""
echo "========== Step 4: 目录整理完成 =========="
echo "📂 临时输出目录: $TEMP_OUTPUT_DIR"
ls -la "$TEMP_OUTPUT_DIR/" | head -10
echo ""

# Step 5: 上传到目标记录
echo "========== Step 5: 上传到指定记录 =========="

if [[ -z "${recordName:-}" ]]; then
  echo "⚠️  未设置 recordName 环境变量，跳过上传"
  echo "💡 如需上传到 CoScene，请设置 recordName 环境变量"
  echo "💡 整理好的目录保存在: $TEMP_OUTPUT_DIR"
else
  echo "目标记录ID: $recordName"

  # 找到所有整理好的子目录
  UPLOAD_DIRS=$(find "$TEMP_OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null)
  UPLOAD_COUNT=$(echo "$UPLOAD_DIRS" | grep -v "^$" | wc -l)

  if [[ $UPLOAD_COUNT -eq 0 ]]; then
    echo "❌ 没有可上传的目录"
    exit 1
  fi

  echo "📊 准备上传 $UPLOAD_COUNT 个目录"
  echo "正在上传文件到记录 $recordName ..."

  # 逐个目录处理：复制到 /cos/outputs 下，然后上传
  SUCCESS_COUNT=0
  FAIL_COUNT=0

  for DIR in $UPLOAD_DIRS; do
    DIR_NAME=$(basename "$DIR")

    # 复制到 /cos/files 目录下（cocli 只能访问这个目录）
    UPLOAD_STAGING="$COS_FILE_VOLUME/$DIR_NAME"
    rm -rf "$UPLOAD_STAGING"
    mkdir -p "$UPLOAD_STAGING"

    echo "  准备: $DIR_NAME/"
    # 复制文件（解引用软链接）
    cp -L "$DIR"/*.bag "$UPLOAD_STAGING/" 2>/dev/null || true
    cp "$DIR"/*.json "$UPLOAD_STAGING/" 2>/dev/null || true

    echo "    文件列表:"
    ls -lh "$UPLOAD_STAGING/" 2>/dev/null | head -5

    # 先删除目标记录中可能存在的同名文件夹（确保完全覆盖而非合并）
    echo "    删除目标记录中的旧文件夹（如存在）..."
    cocli record file delete "$recordName" --files "$DIR_NAME/" -f 2>/dev/null || true

    # 上传这个目录（从 /cos/files 复制到目标记录）
    echo "    上传新文件夹..."
    if cocli record file copy "$COS_RECORDID" "$recordName" --files "$DIR_NAME/" -f 2>&1; then
      echo "    ✅ 上传成功"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      echo "    ❌ 上传失败"
      FAIL_COUNT=$((FAIL_COUNT + 1))
    fi

    # 清理临时目录
    rm -rf "$UPLOAD_STAGING"
  done

  echo ""
  echo "📊 上传结果: 成功 $SUCCESS_COUNT / 失败 $FAIL_COUNT"

  if [[ $SUCCESS_COUNT -gt 0 ]]; then
    echo ""
    echo "📂 上传的目录结构:"
    echo "   $recordName/"
    echo "   ├── xxx/"
    echo "   │   ├── xxx.bag"
    echo "   │   └── xxx.json"
    echo "   └── ..."
  fi

  if [[ $FAIL_COUNT -gt 0 && $SUCCESS_COUNT -eq 0 ]]; then
    echo "❌ 所有上传都失败了"
    rm -rf "$TEMP_OUTPUT_DIR"
    exit 1
  fi
fi

# 清理临时目录
rm -rf "$TEMP_OUTPUT_DIR"

echo ""
echo "🎉 Fetch Metadata 完成！"
echo "💡 下一步: 在汇总 record 上运行 convert_tfds.sh 进行批量转换"
exit 0

