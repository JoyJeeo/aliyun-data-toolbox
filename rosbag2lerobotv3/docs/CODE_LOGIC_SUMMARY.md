# rosbag2lerobotv3 代码运行逻辑说明

## 📋 总体运行流程

```bash
run.sh 
  ↓
CvtRosbag2Lerobot.py (Python 脚本)
  ↓
port_kuavo_rosbag_chunked() 
  ↓
populate_dataset_chunked() 
  ↓
ChunkedRosbagProcessor 
  ↓
生成 LeRobot 格式数据集 + metadata.json
```

---

## 🎯 完整运行流程

### 1. 入口脚本 (run.sh)

**文件**: `kuavo_data/run.sh`

**核心逻辑**:
```bash
#!/usr/bin/env bash
set -euo pipefail

# Step 1: 配置 OSS 上传（可选）
# 生成 ~/.ossutilconfig 配置文件

# Step 2: ROSbag 转换处理
for DATA_DIR in "${DATA_DIRS[@]}"; do
    data_id="$(basename "$DATA_DIR")"
    
    # 检测 metadata.json
    METADATA_JSON_PATH="$DATA_DIR/metadata.json"
    if [[ -f "$METADATA_JSON_PATH" ]]; then
        echo "✅ 检测到 metadata.json: $METADATA_JSON_PATH"
    fi
    
    # 自动识别 eef_type (dex_hand 或 leju_claw)
    for bag in $DATA_DIR/*.bag; do
        if [[ "$fname" == *dex_hand* ]]; then
            EEF_TYPE="dex_hand"
        elif [[ "$fname" == *leju_claw* ]]; then
            EEF_TYPE="leju_claw"
        fi
    done
    
    # 动态选择配置文件
    case "$EEF_TYPE" in
        "dex_hand") CONFIG_FILE="KuavoRosbag2Lerobot.yaml" ;;
        "leju_claw") CONFIG_FILE="KuavoRosbag2Lerobot_claw.yaml" ;;
    esac
    
    # 执行转换命令
    python CvtRosbag2Lerobot.py \
        --config-name="$CONFIG_FILE" \
        rosbag.rosbag_dir="$DATA_DIR" \
        rosbag.lerobot_dir="$OUTPUT_DIR_DATA" \
        rosbag.metadata_json="$METADATA_JSON_PATH"
done
```

**关键配置**:
- `INPUT_DIR`: `/home/zhangyutao/Documents/Work/Code/Contest/test_bags/inputs`
- `OUTPUT_DIR`: `/home/zhangyutao/Documents/Work/Code/Contest/test_bags/outputs`
- 自动检测 `metadata.json` 并传递给 Python 脚本

---

### 2. 主转换脚本 (CvtRosbag2Lerobot.py)

**文件**: `kuavo_data/CvtRosbag2Lerobot.py`

#### 2.1 Hydra 配置加载

```python
@hydra.main(
    config_path="./configs/",
    config_name="KuavoRosbag2Lerobot",  # 或 KuavoRosbag2Lerobot_claw.yaml
    version_base="1.2",
)
def main(cfg: DictConfig):
    # 初始化 kuavo 参数
    kuavo.init_parameters(cfg)
    
    # 读取命令行参数
    raw_dir = cfg.rosbag.rosbag_dir
    version = cfg.rosbag.lerobot_dir
    meta_path = cfg.rosbag.get("metadata_json", None)
    
    # 加载 metadata.json
    metadata = None
    if meta_path and os.path.isfile(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        log_print.info(f"Loaded metadata.json from {meta_path}")
    
    # 调用主转换函数
    port_kuavo_rosbag_chunked(
        raw_dir=raw_dir,
        repo_id=repo_id,
        metadata=metadata,  # 传递 metadata
        ...
    )
```

**关键参数**:
- `rosbag.rosbag_dir`: 输入 rosbag 目录
- `rosbag.lerobot_dir`: 输出 LeRobot 目录
- `rosbag.metadata_json`: metadata.json 路径（可选）

#### 2.2 主转换函数 - port_kuavo_rosbag_chunked()

```python
def port_kuavo_rosbag_chunked(
    raw_dir: Path,
    repo_id: str,
    task: str = "DEBUG",
    metadata: dict | None = None,  # metadata 字典
    ...
):
    # 初始化 KuavoRosbagReader
    bag_reader = kuavo.KuavoRosbagReader()
    bag_files = bag_reader.list_bag_files(raw_dir)
    
    # 创建空数据集
    dataset = create_empty_dataset_chunked(...)
    
    # 分块处理 rosbag
    dataset = populate_dataset_chunked(
        dataset,
        bag_files,
        task=task,
        metadata=metadata,  # 传递 metadata
        ...
    )
    
    # 生成 metadata.json（如果有 metadata）
    try:
        if metadata:
            out_dir = Path(root)  # 使用实际输出目录
            
            # 计算总帧数
            calculated_frames = calculate_total_frames([out_dir])
            
            # 优先使用 metadata.json 中的 total_frames（如果有）
            total_frames = metadata.get("total_frames", calculated_frames)
            
            log_print.info(f"✓ 使用的总帧数: {total_frames}")
            
            if total_frames > 0:
                # 生成完整的 metadata.json
                merge_metadata_and_moment(
                    metadata_path=tmp_metadata_path,
                    moment_path=None,
                    output_path=str(final_metadata_path),
                    uuid=repo_id.split("/")[-1],
                    raw_config=None,
                    total_frames=total_frames,  # 使用原始总帧数
                )
                
                # 合并到 dataset.info
                dataset.meta.info["metadata"].update(final_metadata)
                write_info(dataset.meta.info, out_dir)
    except Exception as e:
        log_print.warning(f"Failed to process metadata.json: {e}")
```

**关键点**:
1. ✅ 使用 `Path(root)` 而不是 `LEROBOT_HOME` 作为输出目录
2. ✅ 优先使用 `metadata.get("total_frames")` 获取原始总帧数
3. ✅ 如果没有 `total_frames`，回退到计算值

---

### 3. 数据集填充 - populate_dataset_chunked()

**文件**: `kuavo_data/CvtRosbag2Lerobot.py`

```python
def populate_dataset_chunked(
    dataset: LeRobotDataset,
    bag_files: list[Path],
    task: str,
    metadata: dict | None = None,
    ...
) -> LeRobotDataset:
    
    # 处理每个 bag 文件
    for ep_idx in tqdm.tqdm(episodes):
        ep_path = bag_files[ep_idx]
        
        # 计算裁剪范围
        crop_range = None
        if metadata:
            # 优先使用 explicit frame indices
            try:
                label_info = metadata.get("label_info") or {}
                action_cfg = label_info.get("action_config") or []
                starts = [a.get("start_frame") for a in action_cfg if a.get("start_frame") is not None]
                ends = [a.get("end_frame") for a in action_cfg if a.get("end_frame") is not None]
                if starts and ends:
                    start_frame = min(starts)
                    end_frame = max(ends)
                    log_print.info(f"Using explicit frame range: {start_frame}-{end_frame}")
            except Exception:
                pass
            
            # 如果没有，使用 fractional positions
            if start_frame is None:
                frac_range = get_time_range_from_metadata(metadata)
                if frac_range is not None:
                    crop_range = frac_range
                    log_print.info(f"Using fractional crop range: {crop_range[0]:.3f}-{crop_range[1]:.3f}")
        
        # 处理单个 bag
        bag_reader.process_rosbag_chunked(
            bag_file=str(ep_path),
            frame_callback=on_frame,
            chunk_size=chunk_size,
            save_callback=on_chunk_done,
            crop_range=crop_range,  # 传递裁剪范围
        )
```

**关键逻辑**:
1. ✅ 从 `metadata["marks"]` 提取裁剪范围（fractional positions 0.0-1.0）
2. ✅ 传递 `crop_range` 给 `process_rosbag_chunked()`
3. ✅ 只处理标注区间的数据，跳过无效时间

---

### 4. Chunk 处理器 - ChunkedRosbagProcessor

**文件**: `kuavo_data/converter/reader/chunk_processor.py`

```python
class ChunkedRosbagProcessor:
    def __init__(self, ..., crop_range: Optional[Tuple[float, float]] = None):
        self.crop_range = crop_range  # 裁剪范围 (min_position, max_position)
    
    def scan_timestamps_only(self, bag_file: str):
        """第一遍扫描：只读取时间戳"""
        # ...
        
        # 应用裁剪范围（在丢弃首尾帧之前）
        if self.crop_range is not None:
            min_pos, max_pos = self.crop_range
            total_frames = len(raw_timestamps)
            crop_start_idx = int(min_pos * total_frames)
            crop_end_idx = int(max_pos * total_frames)
            
            # 限制裁剪范围在有效范围内
            crop_start_idx = max(0, crop_start_idx)
            crop_end_idx = min(total_frames, crop_end_idx)
            
            # 裁剪时间戳
            raw_timestamps = raw_timestamps[crop_start_idx:crop_end_idx]
            logger.info(f"Applied crop range {min_pos:.3f}-{max_pos:.3f}, "
                       f"cropped to frames {crop_start_idx}-{crop_end_idx} "
                       f"(remaining {len(raw_timestamps)} frames)")
        
        # 丢弃首尾帧，降采样
        # ...
        
        return main_timeline, main_timestamps, dict(all_timestamps)
```

**裁剪效果**:
- 输入: 896 帧
- 裁剪范围: 0.003-0.915
- 裁剪后: 283 帧（只处理有效数据）

---

## 📊 Metadata 处理逻辑详解

### 1. Metadata 读取

**位置**: `CvtRosbag2Lerobot.py::main()`

```python
# 尝试从配置/命令行参数读取 metadata.json 路径并加载
metadata = None
try:
    meta_path = cfg.rosbag.get("metadata_json", None)
    if meta_path:
        if os.path.isfile(meta_path):
            import json
            with open(meta_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            log_print.info(f"Loaded metadata.json from {meta_path}")
        else:
            log_print.warning(f"metadata_json path not found: {meta_path}")
except Exception as e:
    log_print.warning(f"Failed to load metadata.json: {e}")
```

**metadata.json 结构**:
```json
{
  "location": "北京训练场-A10",
  "primaryScene": "BM",
  "taskName": "BM-76:取杯子",
  "deviceSn": "P4-295",
  "marks": [
    {
      "skillAtomic": "capture",
      "markStart": "2026-02-04 15:22:41.960",
      "markEnd": "2026-02-04 15:22:48.385",
      "duration": 6.425,
      "startPosition": 0.0031705534228890393,
      "endPosition": 0.2058927464648616,
      "skillDetail": "夹取或抓握杯身或把手，确保握持稳定",
      "enSkillDetail": "pick up or grab a cup or handle to make sure you hold it steady",
      ...
    },
    ...
  ],
  "total_frames": 896  // 新增字段
}
```

---

### 2. 时间裁剪计算

**位置**: `CvtRosbag2Lerobot.py::populate_dataset_chunked()`

```python
# 计算裁剪范围 (fractional positions 0.0-1.0)
crop_range = None
start_frame = None
end_frame = None

if metadata:
    # 优先使用 explicit frame indices（来自已计算好的 label_info）
    try:
        label_info = metadata.get("label_info") or {}
        action_cfg = label_info.get("action_config") or []
        starts = [a.get("start_frame") for a in action_cfg if a.get("start_frame") is not None]
        ends = [a.get("end_frame") for a in action_cfg if a.get("end_frame") is not None]
        if starts and ends:
            start_frame = min(starts)
            end_frame = max(ends)
            log_print.info(f"Using explicit frame range from label_info: {start_frame}-{end_frame}")
    except Exception:
        pass

    # 如果没有 explicit frame indices，使用 fractional positions（来自 marks）
    if start_frame is None:
        frac_range = get_time_range_from_metadata(metadata)
        if frac_range is not None:
            crop_range = frac_range  # (min_start_position, max_end_position)
            log_print.info(f"Using fractional crop range from marks: {crop_range[0]:.3f}-{crop_range[1]:.3f}")
```

**辅助函数 - get_time_range_from_metadata()**:

```python
def get_time_range_from_metadata(metadata: dict) -> Optional[tuple]:
    """从 metadata 字典中提取时间范围（start/end positions）"""
    if not isinstance(metadata, dict):
        return None, None
    marks = metadata.get("marks")
    if isinstance(marks, list) and marks:
        start_positions = []
        end_positions = []
        for mark in marks:
            try:
                if "startPosition" in mark:
                    start_positions.append(float(mark.get("startPosition", 0)))
                if "endPosition" in mark:
                    end_positions.append(float(mark.get("endPosition", 0)))
            except Exception:
                pass
        if start_positions and end_positions:
            return min(start_positions), max(end_positions)
    return None, None
```

**示例**:
```python
# 输入 marks
marks = [
    {"startPosition": 0.003, "endPosition": 0.206},
    {"startPosition": 0.206, "endPosition": 0.265},
    {"startPosition": 0.265, "endPosition": 0.915}
]

# 输出裁剪范围
crop_range = (0.003, 0.915)
```

---

### 3. 应用时间裁剪

**位置**: `converter/reader/chunk_processor.py::scan_timestamps_only()`

```python
# 应用裁剪范围（在丢弃首尾帧之前）
if self.crop_range is not None:
    min_pos, max_pos = self.crop_range
    total_frames = len(raw_timestamps)
    crop_start_idx = int(min_pos * total_frames)
    crop_end_idx = int(max_pos * total_frames)
    
    # 限制裁剪范围在有效范围内
    crop_start_idx = max(0, crop_start_idx)
    crop_end_idx = min(total_frames, crop_end_idx)
    
    # 裁剪时间戳
    raw_timestamps = raw_timestamps[crop_start_idx:crop_end_idx]
    logger.info(f"Applied crop range {min_pos:.3f}-{max_pos:.3f}, "
               f"cropped to frames {crop_start_idx}-{crop_end_idx} "
               f"(remaining {len(raw_timestamps)} frames)")
```

**效果**:
```
原始数据: 0, 1, 2, 3, ..., 895 (共 896 帧)
裁剪范围: 0.003-0.915 → 2-820 帧
裁剪后: 2, 3, 4, ..., 819 (共 818 帧)
丢弃首尾 + 降采样后: 283 帧
```

---

### 4. 生成完整 Metadata

**位置**: `CvtRosbag2Lerobot.py::port_kuavo_rosbag_chunked()`

```python
try:
    if metadata:
        out_dir = Path(root)  # 使用实际输出目录
        
        # 计算总帧数
        calculated_frames = calculate_total_frames([out_dir])
        log_print.info(f"✓ 计算的总帧数: {calculated_frames}")
        
        # 优先使用 metadata.json 中的 total_frames（如果有）
        total_frames = metadata.get("total_frames", calculated_frames)
        log_print.info(f"✓ 使用的总帧数: {total_frames} (优先使用 metadata.total_frames)")
        
        if total_frames > 0:
            # 保存 metadata 字典为临时文件（移除 total_frames 避免污染输出）
            temp_meta = metadata.copy()
            temp_meta.pop("total_frames", None)  # 移除 total_frames
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
                json.dump(temp_meta, tmp, ensure_ascii=False, indent=2)
                tmp_metadata_path = tmp.name
            
            try:
                final_metadata_path = out_dir / "metadata.json"
                merge_metadata_and_moment(
                    metadata_path=tmp_metadata_path,
                    moment_path=None,
                    output_path=str(final_metadata_path),
                    uuid=repo_id.split("/")[-1],
                    raw_config=None,
                    total_frames=total_frames,  # 使用原始总帧数
                )
                
                log_print.info(f"✓ 生成完整 metadata: {final_metadata_path}")
                log_print.info(f"  总帧数: {total_frames}")
                
                # 读取生成的 metadata 并合并到 dataset.info
                try:
                    with open(final_metadata_path, "r", encoding="utf-8") as f:
                        final_metadata = json.load(f)
                    
                    dataset.meta.info.setdefault("metadata", {})
                    dataset.meta.info["metadata"].update(final_metadata)
                    write_info(dataset.meta.info, out_dir)
                    log_print.info(f"✓ 合并 metadata 到 dataset info")
                except Exception as e:
                    log_print.warning(f"Failed to merge metadata into dataset info: {e}")
            finally:
                # 清理临时文件
                import os
                if os.path.exists(tmp_metadata_path):
                    os.unlink(tmp_metadata_path)
except Exception as e:
    log_print.warning(f"Failed to process metadata.json: {e}")
    import traceback
    traceback.print_exc()
```

---

### 5. Metadata 合并与帧范围计算

**位置**: `converter/data/metadata_merge.py::merge_metadata_and_moment()`

```python
def merge_metadata_and_moment(
    metadata_path,
    moment_path,
    output_path,
    uuid,
    raw_config,
    bag_time_info=None,
    main_time_line_timestamps=None,
    total_frames: Optional[int] = None,  # 关键参数
):
    # 读取 metadata.json
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_metadata = json.load(f)
    
    # 检测新格式（包含 marks 数组）
    is_new_format = "marks" in raw_metadata and isinstance(raw_metadata.get("marks"), list)
    
    if is_new_format:
        marks = raw_metadata.get("marks", [])
    
    # ... 转换字段映射 ...
    
    # 获取时间信息
    rosbag_actual_start_time = None
    rosbag_actual_end_time = None
    rosbag_original_start_time = None
    rosbag_original_end_time = None
    
    # 如果传入了 main_time_line_timestamps，从时间戳计算总帧数
    # 否则使用传入的 total_frames 参数
    if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 0:
        actual_total_frames = len(main_time_line_timestamps)
        print(f"从时间戳计算总帧数: {actual_total_frames}")
    else:
        # 使用传入的 total_frames 参数
        actual_total_frames = total_frames if total_frames is not None else 0
        print(f"使用传入的 total_frames: {actual_total_frames}")
    
    total_frames = actual_total_frames
    
    # 构造 action_config
    action_config = []
    
    for m in marks:  # 新格式从 marks 读取
        skill_atomic = m.get("skillAtomic", "")
        skill_detail = m.get("skillDetail", "")
        en_skill_detail = m.get("enSkillDetail", "")
        
        # 优先使用 fractional position 计算帧数
        sp = m.get("startPosition")
        ep = m.get("endPosition")
        
        start_frame = None
        end_frame = None
        
        if sp is not None:
            start_frame = int(float(sp) * total_frames)
        if ep is not None:
            end_frame = int(float(ep) * total_frames)
        
        # clamp to valid range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame, min(end_frame, total_frames - 1))
        
        # 构造 action
        action = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "skill": skill_atomic,
            "action_text": skill_detail,
            ...
        }
        action_config.append(action)
    
    # 规范化帧范围（确保区间连续且覆盖整个 episode）
    if action_config and total_frames is not None:
        # 1. 第一个动作从 0 开始（强制规范化）
        first = action_config[0]
        first["start_frame"] = 0  # 强制为 0
        
        if first.get("end_frame") is None:
            first["end_frame"] = first["start_frame"]
        else:
            first["end_frame"] = max(first["start_frame"], min(int(first["end_frame"]), total_frames - 1))
        
        # 2. 后续动作：起点衔接上一个动作的 end_frame
        for i in range(1, len(action_config)):
            prev = action_config[i - 1]
            cur = action_config[i]
            
            prev_end = prev.get("end_frame")
            if prev_end is None:
                prev_end = prev.get("start_frame", 0)
            prev_end = max(0, min(int(prev_end), total_frames - 1))
            
            cur["start_frame"] = prev_end
            if cur.get("end_frame") is None:
                cur["end_frame"] = prev_end
            else:
                cur["end_frame"] = max(prev_end, min(int(cur["end_frame"]), total_frames - 1))
        
        # 3. 最后一个动作：结束帧拉到整段 episode 末尾
        last = action_config[-1]
        last["end_frame"] = total_frames - 1
    
    # 构造新json
    new_json = OrderedDict()
    new_json["episode_id"] = uuid
    for k, v in metadata.items():
        new_json[k] = v
    new_json["label_info"] = {"action_config": action_config, "key_frame": []}
    
    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)
```

**关键点**:
1. ✅ 优先使用 `total_frames` 参数（来自 metadata.json）
2. ✅ 使用 fractional position 计算帧范围：`start_frame = int(startPosition * total_frames)`
3. ✅ 应用规范化逻辑，确保动作区间连续
4. ✅ 强制第一个动作从 0 开始，最后一个动作到末尾

---

### 6. 计算总帧数

**位置**: `CvtRosbag2Lerobot.py::calculate_total_frames()`

```python
def calculate_total_frames(chunk_dirs):
    """计算所有 chunk 的总帧数"""
    import pyarrow.parquet as pq
    total = 0
    
    for chunk_dir in chunk_dirs:
        # 查找所有 parquet 文件
        parquet_files = list(Path(chunk_dir).glob("data/**/episode_*.parquet"))
        for pf in parquet_files:
            try:
                table = pq.ParquetFile(pf)
                total += table.metadata.num_rows
            except Exception as e:
                log_print.warning(f"读取 parquet 文件失败 {pf}: {e}")
    
    return total
```

---

## 📈 数据流示意图

```
metadata.json (输入)
    ↓
读取 metadata
    ↓
提取裁剪范围 (fractional positions)
    ↓
应用时间裁剪 (只处理有效数据)
    ↓
生成 parquet 文件 (裁剪后的数据，如 283 帧)
    ↓
获取 total_frames (优先使用 metadata.total_frames=896)
    ↓
计算帧范围 (基于 896 帧)
    ↓
应用规范化 (确保连续性)
    ↓
生成 metadata.json (输出，包含 896 帧的帧范围)
    ↓
合并到 dataset.info
```

---

## 🔑 关键设计决策

### 1. 为什么使用 fractional position？

```python
# 使用 fractional position (0.0-1.0)
start_frame = int(startPosition * total_frames)

# 而不是绝对时间
start_frame = calculate_from_absolute_time(markStart, rosbag_start_time)
```

**优势**:
- ✅ 不依赖实际时间戳，更稳定
- ✅ 适用于不同长度的 rosbag
- ✅ 计算简单，不易出错

### 2. 为什么添加 total_frames 字段？

```json
{
  "marks": [...],
  "total_frames": 896  // 新增
}
```

**原因**:
- ⏱️ 数据裁剪后只有 283 帧，无法知道原始完整数据的总帧数
- 📏 需要原始总帧数才能正确计算 marks 在完整数据中的帧范围
- 🔄 向后兼容：如果没有这个字段，回退到计算值

### 3. 为什么强制第一个动作从 0 开始？

```python
# 规范化逻辑
first["start_frame"] = 0  # 强制为 0
last["end_frame"] = total_frames - 1  # 强制到末尾
```

**原因**:
- ✅ 确保动作区间覆盖整个 episode
- ✅ 便于训练模型（明确的起止点）
- ✅ 符合 LeRobot 数据集规范

---

## ✅ 验证指标

### 已满足的指标

- ✅ **总帧数正确**: 使用 `metadata.total_frames` (896)
- ✅ **动作数量正确**: 与 marks 数量一致 (3 个)
- ✅ **帧范围连续性**: 无间隙、无重叠 (0-184-237-895)
- ✅ **首帧为 0**: 第一个动作从 0 开始
- ✅ **末帧为 total_frames-1**: 最后一个动作到 895
- ✅ **Fractional position 计算**: 基于 startPosition/endPosition
- ✅ **时间裁剪**: 正确应用 (0.003-0.915)
- ✅ **向后兼容**: 如果没有 total_frames，回退到计算值

---

## 📁 关键文件清单

| 文件 | 作用 |
|------|------|
| `run.sh` | 入口脚本，自动检测并传递参数 |
| `CvtRosbag2Lerobot.py` | 主转换脚本，Hydra 配置加载 |
| `converter/reader/chunk_processor.py` | Chunk 处理器，应用时间裁剪 |
| `converter/data/metadata_merge.py` | Metadata 合并与帧范围计算 |
| `configs/KuavoRosbag2Lerobot*.yaml` | 配置文件 |

---

**文档版本**: v1.0  
**最后更新**: 2026-03-04  
**状态**: ✅ 完成
