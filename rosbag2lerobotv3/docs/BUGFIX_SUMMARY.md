# 时间裁剪与 Metadata 处理功能修复总结

## 问题描述

在运行 rosbag 转换脚本时，虽然转换成功完成，但生成的 `metadata.json` 中的 `action_config` 的帧范围计算全部为 0-0，无法正确反映实际的动作区间。

## 根本原因

### 问题 1: 导入路径错误
**文件**: `kuavo_data/common/kuavo_dataset.py`
- 尝试从 `common.config_dataset` 导入，但该文件已被移动到 `converter/configs/dataset_config.py`

**修复**: 
```python
# 旧代码
from .config_dataset import load_config

# 新代码
from converter.configs.dataset_config import load_config
```

### 问题 2: 输出目录路径错误
**文件**: `kuavo_data/CvtRosbag2Lerobot.py`
- 在 `port_kuavo_rosbag_chunked` 函数中使用 `LEROBOT_HOME / repo_id` 作为输出目录
- 实际的 parquet 文件保存在 `root` 路径（由 `lerobot_dir` 指定）

**修复**:
```python
# 旧代码
out_dir = Path(LEROBOT_HOME) / repo_id

# 新代码
out_dir = Path(root)
```

### 问题 3: Metadata 参数传递错误
**文件**: `kuavo_data/CvtRosbag2Lerobot.py`
- 调用 `merge_metadata_and_moment` 时传入 `metadata_path=None`
- 但函数内部需要读取文件，导致错误

**修复**:
```python
# 保存 metadata 字典为临时文件
with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
    json.dump(metadata, tmp, ensure_ascii=False, indent=2)
    tmp_metadata_path = tmp.name

try:
    merge_metadata_and_moment(
        metadata_path=tmp_metadata_path,  # 使用临时文件路径
        moment_path=None,
        output_path=str(final_metadata_path),
        uuid=repo_id.split("/")[-1],
        raw_config=None,
        total_frames=total_frames,
    )
finally:
    # 清理临时文件
    import os
    if os.path.exists(tmp_metadata_path):
        os.unlink(tmp_metadata_path)
```

### 问题 4: total_frames 参数被覆盖
**文件**: `kuavo_data/converter/data/metadata_merge.py`
- 函数第 157 行：`total_frames = 0` 覆盖了传入的参数
- 函数假设会传入 `main_time_line_timestamps` 来计算帧数，但实际调用时没有传入

**修复**:
```python
# 旧代码
total_frames = 0
if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 0:
    total_frames = len(main_time_line_timestamps)

# 新代码
if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 0:
    actual_total_frames = len(main_time_line_timestamps)
    print(f"从时间戳计算总帧数: {actual_total_frames}")
else:
    # 使用传入的 total_frames 参数
    actual_total_frames = total_frames if total_frames is not None else 0
    print(f"使用传入的 total_frames: {actual_total_frames}")

total_frames = actual_total_frames
```

## 修复效果

### 修复前
```json
{
  "label_info": {
    "action_config": [
      {
        "skill": "capture",
        "start_frame": 0,
        "end_frame": 0  // ❌ 错误
      },
      {
        "skill": "remove",
        "start_frame": 0,
        "end_frame": 0  // ❌ 错误
      },
      {
        "skill": "placement",
        "start_frame": 0,
        "end_frame": -1  // ❌ 错误
      }
    ]
  }
}
```

### 修复后
```json
{
  "label_info": {
    "action_config": [
      {
        "skill": "capture",
        "start_frame": 0,
        "end_frame": 58   // ✅ 正确
      },
      {
        "skill": "remove",
        "start_frame": 58,
        "end_frame": 75   // ✅ 正确
      },
      {
        "skill": "placement",
        "start_frame": 75,
        "end_frame": 282  // ✅ 正确（283-1）
      }
    ]
  }
}
```

## 验证指标

- ✅ **总帧数计算**: 正确计算为 283 帧
- ✅ **帧范围连续性**: 动作区间无间隙、无重叠
- ✅ **首帧为 0**: 第一个动作从帧 0 开始
- ✅ **末帧为 total_frames-1**: 最后一个动作结束于 282 (283-1)
- ✅ **fractional position 计算**: 使用 startPosition/endPosition 正确计算帧范围
- ✅ **时间裁剪**: 正确应用 (0.003-0.915) 裁剪范围，只处理有效数据

## 关键修复点

1. **模块导入路径**: 更新为新的模块结构
2. **输出目录定位**: 使用实际的 `root` 路径而不是 `LEROBOT_HOME`
3. **参数传递机制**: 将 metadata 字典保存为临时文件再传入
4. **total_frames 处理**: 支持直接传入帧数，而不仅依赖时间戳数组

## 修改的文件

1. `kuavo_data/common/kuavo_dataset.py` - 修复导入路径
2. `kuavo_data/CvtRosbag2Lerobot.py` - 修复输出目录和临时文件机制
3. `kuavo_data/converter/data/metadata_merge.py` - 修复 total_frames 处理逻辑

## 测试命令

```bash
cd rosbag2lerobotv3/kuavo_data
bash run.sh
```

运行成功后，检查输出：
```bash
cat /path/to/output/metadata.json | python3 -m json.tool
```

## 总结

本次修复解决了 metadata.json 处理功能的多个关键问题，使系统能够：
- 正确计算动作的帧范围
- 支持时间裁剪功能
- 生成包含完整 `action_config` 的 metadata
- 确保动作区间连续且覆盖整个 episode
