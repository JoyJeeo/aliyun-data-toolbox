# ROSbag2LeRobotV3 项目深度研究报告

## 1. 项目概述

**项目定位**: 轻量级、内存优化的 Kuavo 机器人 ROSbag 到 LeRobot 数据集转换系统

**核心目标**: 以最小内存占用实现 ROSbag 到 LeRobot 格式的转换，特别适合资源受限环境和超大 bag 文件处理。

**技术特点**:
- 分块流式处理，内存占用可控
- 简化架构，易于理解和维护
- Hydra 配置管理，灵活便捷
- 集成 metadata 处理和批次合并

---

## 2. 系统架构

### 2.1 整体架构设计

```
CvtRosbag2Lerobot.py (主入口)
    ↓
port_kuavo_rosbag_chunked (转换函数)
    ↓
├── KuavoRosbagReader (数据读取器)
│   ├── scan_timestamps_only (第一遍扫描)
│   └── process_in_chunks (第二遍分块处理)
│
├── ChunkedRosbagProcessor (分块处理器)
│   ├── scan_timestamps_only (时间戳扫描)
│   ├── process_in_chunks (分块处理)
│   └── _align_single_frame (单帧对齐)
│
├── metadata_merge (元数据处理)
│   ├── merge_metadata_and_moment (合并元数据)
│   └── get_time_range_from_metadata (提取时间范围)
│
└── batch_merger (批次合并)
    └── merge_all_batches (合并所有批次)
```

### 2.2 核心设计理念

**两遍扫描策略** (参考 Diffusion Policy):
1. **第一遍**: 只读取时间戳，确定主时间线（内存占用几 MB）
2. **第二遍**: 按时间窗口分块读取+对齐+写入（内存可控）

**与 V21 的区别**:
- V21: 一次性加载所有数据 → 对齐 → 写入（内存峰值巨大）
- V3: 分块读取 → 即时对齐 → 即时写入 → 释放内存（内存可控）

---

## 3. 核心功能深度解析

### 3.1 分块流式处理系统

#### 3.1.1 第一遍扫描：时间戳收集

```python
# chunk_processor.py
def scan_timestamps_only(self, bag_file):
    """
    第一遍扫描：只读取时间戳，不加载数据
    内存占用：只有时间戳列表（几MB），不包含图像数据
    """
    bag = self._load_bag(bag_file)
    all_timestamps = defaultdict(list)

    # 一次遍历，收集所有话题的时间戳
    for topic, msg, t in bag.read_messages(topics=all_topics):
        keys = topic_to_key.get(topic)
        for key in keys:
            all_timestamps[key].append(t.to_sec())

    # 确定主时间线：消息最多的相机
    main_timeline = max(camera_names, key=lambda k: len(all_timestamps.get(k, [])))

    # 生成对齐后的主时间戳序列
    jump = self.main_timeline_fps // self.train_hz
    raw_timestamps = all_timestamps[main_timeline]

    # 应用裁剪范围（如果有）
    if self.crop_range is not None:
        min_pos, max_pos = self.crop_range
        total_frames = len(raw_timestamps)
        crop_start_idx = int(min_pos * total_frames)
        crop_end_idx = int(max_pos * total_frames)
        raw_timestamps = raw_timestamps[crop_start_idx:crop_end_idx]

    # 丢弃首尾帧，降采样
    if self.sample_drop > 0:
        main_timestamps = raw_timestamps[self.sample_drop:-self.sample_drop][::jump]
    else:
        main_timestamps = raw_timestamps[::jump]

    return main_timeline, main_timestamps, dict(all_timestamps)
```

**关键优化**:
- 只读时间戳，不解析消息内容
- 内存占用极小（几 MB）
- 支持裁剪范围（crop_range）

#### 3.1.2 第二遍扫描：分块处理

```python
# chunk_processor.py
def process_in_chunks(
    self,
    bag_file,
    main_timestamps,
    all_timestamps,
    frame_callback,
    chunk_size=100,
    save_callback=None
):
    """
    第二遍扫描：按时间窗口分块处理

    策略：
    1. 将 main_timestamps 分成多个 chunk
    2. 对于每个 chunk，只读取该时间范围内的消息
    3. 对齐后立即调用 frame_callback
    4. 每个 chunk 处理完后调用 save_callback 释放内存
    """
    bag = self._load_bag(bag_file)

    # 预计算每个主时间戳对应的各话题索引（避免重复查找）
    alignment_indices = self._precompute_alignment_indices(
        main_timestamps, timestamp_arrays
    )

    num_chunks = (len(main_timestamps) + chunk_size - 1) // chunk_size

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(main_timestamps))
        chunk_timestamps = main_timestamps[start_idx:end_idx]

        # 确定该 chunk 的时间范围（扩展一点以确保对齐数据可用）
        time_margin = 1.0 / self.train_hz
        chunk_start_time = chunk_timestamps[0] - time_margin
        chunk_end_time = chunk_timestamps[-1] + time_margin

        # 读取该时间范围内的消息
        chunk_data = self._read_chunk_data(bag, chunk_start_time, chunk_end_time)

        # 对齐并处理每帧
        for global_idx, main_stamp in enumerate(chunk_timestamps, start=start_idx):
            aligned_frame = self._align_single_frame(
                main_stamp, global_idx, chunk_data,
                timestamp_arrays, alignment_indices
            )
            frame_callback(aligned_frame, global_idx)

        # 释放 chunk 数据
        del chunk_data

        # 调用保存回调
        if save_callback:
            save_callback()
```

**关键优化**:
1. **时间窗口读取**: 只读取当前 chunk 需要的数据
2. **预计算索引**: 使用二分查找预计算对齐索引
3. **即时处理**: 对齐后立即回调，不积累数据
4. **及时释放**: 每个 chunk 处理完立即释放内存

#### 3.1.3 单帧对齐

```python
# chunk_processor.py
def _align_single_frame(
    self,
    main_stamp,
    global_idx,
    chunk_data,
    timestamp_arrays,
    alignment_indices
):
    """对齐单帧数据"""
    aligned_frame = {"timestamp": main_stamp}

    for key in self._topic_process_map.keys():
        # 获取预计算的索引
        if key not in alignment_indices or global_idx >= len(alignment_indices[key]):
            aligned_frame[key] = None
            continue

        closest_idx = alignment_indices[key][global_idx]
        target_ts = timestamp_arrays[key][closest_idx]

        # 从 chunk_data 中查找数据
        if key in chunk_data:
            ts_list = list(chunk_data[key].keys())
            if ts_list:
                closest_chunk_ts = min(ts_list, key=lambda x: abs(x - target_ts))
                aligned_frame[key] = chunk_data[key][closest_chunk_ts]
            else:
                aligned_frame[key] = None
        else:
            aligned_frame[key] = None

    return aligned_frame
```

### 3.2 主入口与数据流

```python
# CvtRosbag2Lerobot.py
def populate_dataset_chunked(
    dataset,
    bag_files,
    task,
    chunk_size=800,
    metadata=None
):
    """使用分块流式处理填充数据集"""

    bag_reader = kuavo.KuavoRosbagReader()

    for ep_idx, ep_path in enumerate(bag_files):
        # 计算裁剪范围
        crop_range = None
        if metadata:
            crop_range = get_time_range_from_metadata(metadata)

        # 收集当前 episode 的所有帧
        frames_buffer = []
        frame_count = [0]

        def on_frame(aligned_frame, frame_idx):
            """处理单帧对齐数据"""
            # 1. 提取 state / action
            state = get_array('observation.state', np.float32)
            action = get_array('action', np.float32)

            # 2. 处理 arm trajectory
            arm_traj_alt = get_array("action.kuavo_arm_traj_alt", np.float32)
            if arm_traj_alt.size:
                action[12:26] = arm_traj_alt

            # 3. 处理手部数据（claw/qiangnao/rq2f85）
            # ... 归一化和拼接

            # 4. 构建 frame
            frame = {
                "observation.state": torch.from_numpy(final_state),
                "action": torch.from_numpy(final_action),
            }

            # 5. 添加相机数据
            for cam_key in kuavo.DEFAULT_CAMERA_NAMES:
                cam_data = aligned_frame.get(cam_key)
                if cam_data and "data" in cam_data:
                    frame[f"observation.images.{cam_key}"] = cam_data["data"]

            frames_buffer.append(frame)
            frame_count[0] += 1

        def on_chunk_done():
            """每个 chunk 处理完后的回调：保存并释放内存"""
            if len(frames_buffer) == 0:
                return

            # 将所有缓存的帧添加到 dataset
            for frame in frames_buffer:
                dataset.add_frame(frame, task=task)

            # 保存当前 chunk
            dataset.save_episode()
            dataset.hf_dataset = dataset.create_hf_dataset()

            # 清空 buffer 并释放内存
            frames_buffer.clear()
            gc.collect()

        # 使用分块流式处理
        bag_reader.process_rosbag_chunked(
            bag_file=str(ep_path),
            frame_callback=on_frame,
            chunk_size=chunk_size,
            save_callback=on_chunk_done,
            crop_range=crop_range
        )
```

---

## 4. 配置系统

### 4.1 Hydra 配置管理

```yaml
# KuavoRosbag2Lerobot.yaml
hydra:
  run:
    dir: ./outputs/data_cvt_hydra_save/singlerun/${now:%Y%m%d_%H%M%S}

rosbag:
  rosbag_dir: /path/to/your/rosbag
  metadata_json: null
  num_used: null
  lerobot_dir: /your/path/to/your/lerobotdata/
  chunk_size: 800

dataset:
  only_arm: true
  eef_type: qiangnao  # leju_claw, qiangnao, rq2f85
  which_arm: both  # left, right, both
  use_depth: false
  depth_range: [0, 1500]

  task_description: "Pick and Place"

  train_hz: 10
  main_timeline: head_cam_h
  main_timeline_fps: 30
  sample_drop: 10

  dex_dof_needed: 1
  is_binary: false
  delta_action: false
  relative_start: false

  resize:
    width: 848
    height: 480
```

### 4.2 配置加载

```python
# dataset_config.py
@dataclass
class Config:
    only_arm: bool
    eef_type: str
    which_arm: str
    use_depth: bool
    depth_range: tuple[int, int]
    train_hz: int
    main_timeline: str
    main_timeline_fps: int
    sample_drop: int
    resize: ResizeConfig

    @property
    def use_leju_claw(self) -> bool:
        return "claw" in self.eef_type or self.eef_type == "rq2f85"

    @property
    def use_qiangnao(self) -> bool:
        return self.eef_type == 'qiangnao'

    @property
    def default_camera_names(self) -> List[str]:
        cameras = {
            "left": ['head_cam_h', 'wrist_cam_l'],
            "right": ['head_cam_h', 'wrist_cam_r'],
            "both": ['head_cam_h', 'wrist_cam_l', 'wrist_cam_r']
        }
        if self.use_depth:
            cameras = {
                "left": ['head_cam_h', 'depth_h', 'wrist_cam_l', 'depth_l'],
                "right": ['head_cam_h', 'depth_h', 'wrist_cam_r', 'depth_r'],
                "both": ['head_cam_h', 'depth_h', 'wrist_cam_l', 'depth_l', 'wrist_cam_r', 'depth_r']
            }
        return cameras[self.which_arm]
```

**优势**:
- 使用 `@property` 自动计算派生配置
- 类型安全，IDE 友好
- 配置验证内置

---

## 5. Metadata 处理系统

### 5.1 新旧格式兼容

```python
# metadata_merge.py
def merge_metadata_and_moment(
    metadata_path,
    moment_path,
    output_path,
    uuid,
    raw_config,
    total_frames
):
    """
    合并 metadata 和 moment 数据
    支持两种格式：
    1. 旧格式：metadata.json + moments.json 两个文件
    2. 新格式：只有一个 metadata.json，包含 marks 数组
    """

    # 读取 metadata.json
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_metadata = json.load(f)

    # 检测新格式：如果 metadata.json 中有 marks 字段
    is_new_format = "marks" in raw_metadata and isinstance(raw_metadata.get("marks"), list)

    if is_new_format:
        marks = raw_metadata.get("marks", [])
        # 新格式字段映射
        converted_metadata["scene_name"] = raw_metadata.get("primaryScene", "")
        converted_metadata["task_name"] = raw_metadata.get("taskGroupName", "")
        # ...
    else:
        # 旧格式：读取 moments.json
        with open(moment_path, "r", encoding="utf-8") as f:
            moment = json.load(f)
        # 旧格式字段映射
        converted_metadata["scene_name"] = raw_metadata.get("scene_code", "")
        # ...
```

### 5.2 动作帧计算

```python
# metadata_merge.py
def calculate_action_frames(
    rosbag_actual_start_time,
    rosbag_actual_end_time,
    action_original_start_time,
    action_duration,
    total_frames
):
    """计算动作在数据集中的起止帧"""

    # 裁剪到实际数据范围
    clipped_action_start = max(action_start_time, rosbag_actual_start_time)
    clipped_action_end = min(action_end_time, rosbag_actual_end_time)

    # 计算相对偏移
    start_offset = clipped_action_start - rosbag_actual_start_time
    end_offset = clipped_action_end - rosbag_actual_start_time
    actual_data_duration = rosbag_actual_end_time - rosbag_actual_start_time

    # 转换为帧索引
    start_frame = int((start_offset / actual_data_duration) * total_frames)
    end_frame = int((end_offset / actual_data_duration) * total_frames)

    return start_frame, end_frame
```

### 5.3 Fractional Position 支持

```python
# 新格式支持 fractional position (0.0-1.0)
if is_new_format and total_frames is not None:
    sp = mark.get("startPosition")  # 0.1
    ep = mark.get("endPosition")    # 0.9
    if sp is not None:
        start_frame = int(float(sp) * total_frames)
    if ep is not None:
        end_frame = int(float(ep) * total_frames)
```

**优势**: 不依赖绝对时间戳，更加稳定可靠

---

## 6. 批次合并系统

### 6.1 批次生成

```python
# CvtRosbag2Lerobot.py
def on_chunk_done():
    """每个 chunk 处理完后的回调"""
    # 记录当前 chunk 的帧范围
    chunk_start_frame = frame_count[0] - len(frames_buffer)
    chunk_end_frame = frame_count[0] - 1
    chunk_frame_ranges.append((chunk_start_frame, chunk_end_frame))
    chunk_idx = len(chunk_frame_ranges) - 1

    # 如果有 metadata，保存该 batch 的 metadata
    if metadata and total_frames > 0:
        # 提取该 chunk 范围内的 marks
        chunk_marks = extract_actions_for_chunk(
            metadata, chunk_start_frame, chunk_end_frame, total_frames
        )

        if chunk_marks:
            # 创建 batch 目录
            batch_dir = Path(root) / f"batch_{chunk_idx:04d}"
            batch_dir.mkdir(parents=True, exist_ok=True)

            # 保存该 chunk 的 metadata
            save_chunk_metadata(batch_dir / "metadata.json", metadata, chunk_marks, ...)

            # 复制 data 和 meta 目录到 batch 目录
            shutil.copytree(chunk_data_dir, batch_data_dir)
            shutil.copytree(chunk_meta_dir, batch_meta_out_dir)
```

### 6.2 批次合并

```python
# batch_merger.py
def merge_all_batches(input_dir, output_dir):
    """合并所有 batch"""
    batch_dirs = get_batch_dirs(input_dir)

    # 1. 合并 parquet 文件
    total_frames = merge_parquet_files(batch_dirs, output_dir)

    # 2. 合并 metadata.json
    merged_metadata = merge_batch_metadata(batch_dirs, output_dir, total_frames)

    # 3. 合并 meta 文件
    merge_meta_files(batch_dirs, output_dir, total_frames)

    # 4. 清理 batch 目录
    for batch_dir in batch_dirs:
        shutil.rmtree(batch_dir)
```

### 6.3 动作区间规范化

```python
# metadata_merge.py
# 规范化帧范围（确保区间连续且覆盖整个 episode）
if action_config and total_frames is not None:
    # 1. 第一个动作从 0 开始
    first = action_config[0]
    first["start_frame"] = 0
    first["end_frame"] = max(first["start_frame"], min(int(first["end_frame"]), total_frames - 1))

    # 2. 后续动作：起点衔接上一个动作的 end_frame
    for i in range(1, len(action_config)):
        prev = action_config[i - 1]
        cur = action_config[i]
        cur["start_frame"] = prev["end_frame"]
        cur["end_frame"] = max(prev["end_frame"], min(int(cur["end_frame"]), total_frames - 1))

    # 3. 最后一个动作：结束帧拉到整段 episode 末尾
    last = action_config[-1]
    last["end_frame"] = total_frames - 1
```

---

## 7. 特殊处理

### 7.1 时间裁剪支持

```python
# chunk_processor.py
def scan_timestamps_only(self, bag_file):
    # 应用裁剪范围（在丢弃首尾帧之前）
    if self.crop_range is not None:
        min_pos, max_pos = self.crop_range  # (0.1, 0.9)
        total_frames = len(raw_timestamps)
        crop_start_idx = int(min_pos * total_frames)
        crop_end_idx = int(max_pos * total_frames)

        # 裁剪时间戳
        raw_timestamps = raw_timestamps[crop_start_idx:crop_end_idx]
        logger.info(f"Applied crop range {min_pos:.3f}-{max_pos:.3f}, "
                   f"cropped to frames {crop_start_idx}-{crop_end_idx}")
```

### 7.2 手部数据处理

```python
# CvtRosbag2Lerobot.py
def on_frame(aligned_frame, frame_idx):
    # 读取手部数据
    claw_state = get_array("observation.claw", np.float64)
    qiangnao_state = get_array("observation.qiangnao", np.float64)
    rq2f85_state = get_array("observation.rq2f85", np.float64)

    # 归一化
    if kuavo.IS_BINARY:
        qiangnao_state = np.where(qiangnao_state > 50, 1, 0)
        claw_state = np.where(claw_state > 50, 1, 0)
        rq2f85_state = np.where(rq2f85_state > 0.4, 1, 0)
    else:
        if claw_state.size: claw_state /= 100
        if qiangnao_state.size: qiangnao_state /= 100
        if rq2f85_state.size: rq2f85_state /= 0.8

    # 回退机制
    if claw_action.size == 0 and qiangnao_action.size == 0:
        claw_action = rq2f85_action
        claw_state = rq2f85_state
```

### 7.3 深度图处理

```python
# CvtRosbag2Lerobot.py
if "depth" in cam_key:
    min_d, max_d = kuavo.DEPTH_RANGE
    depth = np.clip(img, min_d, max_d)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-9)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    frame[f"observation.{cam_key}"] = depth_uint8[..., None].repeat(3, -1)
```

---

## 8. 性能优化

### 8.1 内存优化

1. **两遍扫描**: 第一遍只读时间戳，内存占用极小
2. **分块处理**: 每次只处理 chunk_size 帧
3. **即时释放**: 每个 chunk 处理完立即释放
4. **垃圾回收**: 关键节点手动 `gc.collect()`

### 8.2 计算优化

1. **预计算索引**: 使用二分查找预计算对齐索引
2. **向量化操作**: NumPy 向量化替代循环
3. **避免重复解析**: 第一遍扫描缓存时间戳

### 8.3 IO 优化

1. **时间窗口读取**: 只读取当前 chunk 需要的数据
2. **批量写入**: 批次级别保存，减少 IO 次数

---

## 9. 项目优势

### 9.1 技术优势

1. **内存占用可控**: 分块处理，适合超大 bag 文件
2. **架构简洁**: 代码量少，易于理解和维护
3. **配置灵活**: Hydra 配置管理，支持命令行覆盖
4. **格式兼容**: 支持新旧 metadata 格式

### 9.2 使用优势

1. **上手简单**: 架构清晰，学习曲线平缓
2. **资源友好**: 适合资源受限环境
3. **快速迭代**: 代码简洁，易于修改和扩展

---

## 10. 项目不足与改进方向

### 10.1 当前不足

1. **功能相对简单**: 缺少并行处理、流式编码等高级功能
2. **错误处理不足**: 缺少完善的异常处理和容错机制
3. **质量验证缺失**: 缺少数据质量检查和验证
4. **文档不足**: 缺少详细的使用文档

### 10.2 改进方向

1. **增强功能**: 添加并行处理、流式编码支持
2. **完善错误处理**: 增加异常捕获和重试机制
3. **增加质量检查**: 添加数据质量验证和报告
4. **完善文档**: 增加使用示例和 API 文档
5. **性能优化**: 进一步优化内存和计算性能

---

## 11. 与 V21 的对比

| 特性 | V21 | V3 |
|------|-----|-----|
| **架构复杂度** | 高（多层模块） | 低（简化架构） |
| **内存占用** | 较高 | 极低 |
| **处理速度** | 快（并行优化） | 中等 |
| **功能完整性** | 完善 | 基础 |
| **配置系统** | JSON + 环境变量 | Hydra YAML |
| **错误处理** | 完善 | 基础 |
| **学习曲线** | 陡峭 | 平缓 |
| **适用场景** | 生产环境、大规模 | 开发环境、资源受限 |

---

## 12. 总结

ROSbag2LeRobotV3 是一个**轻量级、内存优化**的数据转换系统，适合快速开发和资源受限环境。其核心优势在于：

1. **极低内存占用**: 分块流式处理，适合超大文件
2. **简洁架构**: 代码清晰，易于理解和维护
3. **灵活配置**: Hydra 配置管理，使用便捷
4. **格式兼容**: 支持新旧 metadata 格式

适用场景：
- 资源受限环境（内存<8GB）
- 超大 bag 文件（>50GB）
- 快速原型开发
- 学习和研究

不适用场景：
- 需要高性能处理（无并行支持）
- 生产环境部署（功能相对简单）
- 需要完善错误处理（容错机制不足）

**推荐使用策略**:
- 开发阶段使用 V3（快速迭代）
- 生产环境使用 V21（稳定可靠）
- 资源受限使用 V3（内存优化）
- 大规模处理使用 V21（并行加速）
