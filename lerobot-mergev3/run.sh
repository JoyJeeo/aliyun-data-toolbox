#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

INPUT_DIR="${INPUT_DIR:-/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/outputs}"
OUTPUT_REPO_ID="${OUTPUT_REPO_ID:-lerobot/kuavo-merged-dataset}"
MERGE_SCRIPT="${MERGE_SCRIPT:-}"
DEBUG_SLEEP_SECONDS="${DEBUG_SLEEP_SECONDS:-0}"
TARGET_SCRIPT_NAME="${TARGET_SCRIPT_NAME:-}"
TARGET_SCRIPT_NAMES="${TARGET_SCRIPT_NAMES:-}"
STAGING_MODE="${STAGING_MODE:-copy}"  # copy|symlink ; readonly input should use copy
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_merge_script() {
    if [[ -n "$MERGE_SCRIPT" ]]; then
        if [[ -f "$MERGE_SCRIPT" ]]; then
            return 0
        fi
        if [[ -f "$SCRIPT_DIR/$MERGE_SCRIPT" ]]; then
            MERGE_SCRIPT="$SCRIPT_DIR/$MERGE_SCRIPT"
            return 0
        fi
    fi

    if [[ -f "$SCRIPT_DIR/merge_data.py" ]]; then
        MERGE_SCRIPT="$SCRIPT_DIR/merge_data.py"
        return 0
    fi

    echo "❌ 未找到 merge_data.py，请设置 MERGE_SCRIPT。"
    return 1
}

validate_merged_output() {
    local merged_dir="$1"
    if [[ ! -d "$merged_dir/meta" ]]; then
        echo "❌ 合并结果缺失目录: $merged_dir/meta"
        return 1
    fi
    if [[ ! -d "$merged_dir/data" ]]; then
        echo "❌ 合并结果缺失目录: $merged_dir/data"
        return 1
    fi
    return 0
}

parse_target_scripts() {
    local raw="$1"
    local normalized
    normalized="$(echo "$raw" | tr ',' ' ')"
    read -r -a TARGET_SCRIPTS <<< "$normalized"
}

collect_search_roots() {
    local input_root="$1"
    shift
    local script_names=("$@")
    local data_id_dir script_dir script_name

    while IFS= read -r -d '' data_id_dir; do
        for script_name in "${script_names[@]}"; do
            script_dir="$data_id_dir/$script_name"
            if [[ -d "$script_dir" ]]; then
                printf '%s\0' "$script_dir"
            fi
        done
    done < <(find "$input_root" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)
}

is_v3_dataset_dir() {
    local dir="$1"
    [[ -d "$dir/meta" ]] \
        && [[ -d "$dir/data" ]] \
        && [[ -d "$dir/videos" ]]
}

print_dataset_summary() {
    local dir="$1"
    local parquet_count video_count

    parquet_count="$(find "$dir/data" -type f -name 'episode_*.parquet' | wc -l | tr -d ' ')"
    video_count="$(find "$dir/videos" -type f \( -name '*.mp4' -o -name '*.avi' \) | wc -l | tr -d ' ')"
    echo "    ✅ dataset: $dir"
    echo "       parquet_files=$parquet_count videos=$video_count"
}

find_v3_dataset_dirs() {
    local root="$1"
    local dir

    while IFS= read -r -d '' dir; do
        if is_v3_dataset_dir "$dir"; then
            printf '%s\0' "$dir"
        fi
    done < <(find "$root" -type d -print0 2>/dev/null)
}

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "❌ INPUT_DIR 不存在或不是目录: $INPUT_DIR"
    exit 1
fi

if [[ -n "$TARGET_SCRIPT_NAME" ]]; then
    TARGET_SCRIPT_NAMES="$TARGET_SCRIPT_NAME"
fi
if [[ -z "${TARGET_SCRIPT_NAMES// }" ]]; then
    echo "❌ 缺失 TARGET_SCRIPT_NAME（或 TARGET_SCRIPT_NAMES）。"
    exit 1
fi

if ! resolve_merge_script; then
    exit 1
fi
echo "🧩 使用合并脚本: $MERGE_SCRIPT"
echo "🧩 staging 模式: $STAGING_MODE"

mkdir -p "$OUTPUT_DIR"
if [[ ! -w "$OUTPUT_DIR" ]]; then
    echo "❌ OUTPUT_DIR 不可写: $OUTPUT_DIR"
    exit 1
fi

echo "🔍 递归查找 $INPUT_DIR 下可合并的源目录..."
echo "📁 $INPUT_DIR 下的全部目录列表："
find "$INPUT_DIR" -type d | sort

if [[ "$DEBUG_SLEEP_SECONDS" =~ ^[0-9]+$ ]] && [[ "$DEBUG_SLEEP_SECONDS" -gt 0 ]]; then
    echo "⏸️ 调试暂停 ${DEBUG_SLEEP_SECONDS} 秒，可进入容器检查目录结构..."
    sleep "$DEBUG_SLEEP_SECONDS"
fi

TARGET_SCRIPTS=()
parse_target_scripts "$TARGET_SCRIPT_NAMES"

search_roots=()
echo "🎯 限定脚本目录: ${TARGET_SCRIPTS[*]}"
mapfile -d '' -t search_roots < <(collect_search_roots "$INPUT_DIR" "${TARGET_SCRIPTS[@]}" | sort -z)
if [[ ${#search_roots[@]} -eq 0 ]]; then
    echo "❌ 未找到指定脚本目录: ${TARGET_SCRIPTS[*]}"
    exit 1
fi

echo "🔎 实际检索根目录数量: ${#search_roots[@]}"
for r in "${search_roots[@]}"; do
    echo "  - $r"
done

all_dataset_dirs=()
for root in "${search_roots[@]}"; do
    echo "🔬 扫描 v3 数据集目录: $root"
    while IFS= read -r -d '' dataset_dir; do
        all_dataset_dirs+=("$dataset_dir")
    done < <(find_v3_dataset_dirs "$root")
done

if [[ ${#all_dataset_dirs[@]} -eq 0 ]]; then
    echo "❌ 未找到 v3 数据集目录。要求目录下至少存在 meta/、data/、videos/。"
    exit 1
fi

mapfile -d '' -t all_dataset_dirs < <(printf '%s\0' "${all_dataset_dirs[@]}" | sort -zu)
echo "📦 收集到 ${#all_dataset_dirs[@]} 个 v3 数据集目录，将执行一次全局合并。"
for d in "${all_dataset_dirs[@]}"; do
    print_dataset_summary "$d"
done

staging_dir="$(mktemp -d /tmp/lerobot_merge_v3_staging.XXXXXX)"
cleanup_staging() {
    chmod -R u+w "$staging_dir" 2>/dev/null || true
    rm -rf "$staging_dir"
}
trap cleanup_staging EXIT

idx=1
for d in "${all_dataset_dirs[@]}"; do
    dataset_root="$staging_dir/dataset_$(printf '%04d' "$idx")"
    target="$dataset_root/lerobot"
    mkdir -p "$dataset_root"
    if [[ "$STAGING_MODE" == "symlink" ]]; then
        ln -s "$d" "$target"
    else
        cp -a "$d" "$target"
        chmod -R u+w "$target" 2>/dev/null || true
    fi
    idx=$((idx + 1))
done

merged_dir="$OUTPUT_DIR/lerobot_merged"
echo "========== 全局合并 =========="
echo "汇总目录: $staging_dir"
echo "输出目录: $merged_dir"
echo "输出 repo_id: $OUTPUT_REPO_ID"

rm -rf "$merged_dir"
mkdir -p "$(dirname "$merged_dir")"

if python3 "$MERGE_SCRIPT" \
    --input_dir "$staging_dir" \
    --output_dir "$merged_dir" \
    --output_repo_id "$OUTPUT_REPO_ID"; then
    if validate_merged_output "$merged_dir"; then
        echo "✅ 全局合并完成: $merged_dir"
    else
        echo "❌ 合并结果校验失败。"
        exit 1
    fi
else
    echo "❌ 全局合并失败。"
    exit 1
fi

echo "🎉 全部处理完成，输出目录: $merged_dir"
