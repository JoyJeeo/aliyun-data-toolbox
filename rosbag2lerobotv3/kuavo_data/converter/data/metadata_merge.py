"""Metadata and moments merge helpers."""

from collections import OrderedDict
from typing import Optional, List
import datetime
import json
import os


def calculate_action_frames(
    rosbag_actual_start_time,
    rosbag_actual_end_time,
    rosbag_original_start_time,
    rosbag_original_end_time,
    action_original_start_time,
    action_duration,
    frame_rate,
    total_frames,
):
    action_start_time = action_original_start_time
    action_end_time = action_original_start_time + action_duration

    if (
        action_end_time < rosbag_actual_start_time
        or action_start_time > rosbag_actual_end_time
    ):
        return None, None

    clipped_action_start = max(action_start_time, rosbag_actual_start_time)
    clipped_action_end = min(action_end_time, rosbag_actual_end_time)
    start_offset = clipped_action_start - rosbag_actual_start_time
    end_offset = clipped_action_end - rosbag_actual_start_time
    actual_data_duration = rosbag_actual_end_time - rosbag_actual_start_time

    start_frame = int((start_offset / actual_data_duration) * total_frames)
    end_frame = int((end_offset / actual_data_duration) * total_frames)

    start_frame = max(0, min(start_frame, total_frames - 1))
    end_frame = max(start_frame, min(end_frame, total_frames - 1))
    return start_frame, end_frame


def merge_metadata_and_moment(
    metadata_path,
    moment_path,
    output_path,
    uuid,
    raw_config,
    bag_time_info=None,
    main_time_line_timestamps=None,
    total_frames: Optional[int] = None,
):
    """
    合并 metadata 和 moment 数据，并添加 bag 时间信息和计算帧数
    支持两种格式：
    1. 旧格式：metadata.json + moments.json 两个文件
    2. 新格式：只有一个 metadata.json，包含 marks 数组
    
    Args:
        metadata_path: metadata.json 文件路径
        moment_path: moment.json 文件路径（新格式下可为 None）
        output_path: 输出文件路径
        uuid: 唯一标识符
        raw_config: 原始配置对象
        bag_time_info: bag时间信息字典（可选）
        main_time_line_timestamps: 经过帧率对齐后的时间戳数组（纳秒）
    """
    frequency = raw_config.train_hz if hasattr(raw_config, "train_hz") else 30

    # 读取 metadata.json
    with open(metadata_path, "r", encoding="utf-8") as f:
        raw_metadata = json.load(f)

    # 检测新格式：如果 metadata.json 中有 marks 字段，使用新格式
    is_new_format = "marks" in raw_metadata and isinstance(raw_metadata.get("marks"), list)
    
    if is_new_format:
        print("[FORMAT] 检测到新格式 metadata.json（包含 marks 数组）")
        marks = raw_metadata.get("marks", [])
        moment = None  # 新格式不需要 moment.json
    else:
        print("[FORMAT] 使用旧格式（metadata.json + moments.json）")
        # 读取 moment.json
        if moment_path and os.path.exists(moment_path):
            with open(moment_path, "r", encoding="utf-8") as f:
                moment = json.load(f)
        else:
            print(f"[WARN] moment.json 不存在: {moment_path}")
            moment = {"moments": []}

    # 转换新格式 metadata 为旧格式
    converted_metadata = {}
    
    if is_new_format:
        # 新格式字段映射
        converted_metadata["scene_name"] = raw_metadata.get("primaryScene", "")
        converted_metadata["sub_scene_name"] = raw_metadata.get("tertiaryScene", "")
        converted_metadata["init_scene_text"] = raw_metadata.get("initSceneText", "")
        converted_metadata["english_init_scene_text"] = raw_metadata.get("englishInitSceneText", "")
        
        # task_name 优先 taskGroupName，其次 taskName
        task_name = raw_metadata.get("taskGroupName")
        if not task_name:
            task_name = raw_metadata.get("taskName", "")
        converted_metadata["task_name"] = task_name
        
        # english_task_name 优先 taskGroupCode，其次 taskCode
        english_task_name = raw_metadata.get("taskGroupCode")
        if not english_task_name:
            english_task_name = raw_metadata.get("taskCode", "")
        converted_metadata["english_task_name"] = english_task_name
        if isinstance(english_task_name, str) and "_" in english_task_name:
            english_task_name = english_task_name.replace("_", " ")
        converted_metadata["english_task_name"] = english_task_name
        
        converted_metadata["sn_code"] = raw_metadata.get("deviceSn", "")
    else:
        # 旧格式字段映射
        converted_metadata["scene_name"] = raw_metadata.get("scene_code", "")
        converted_metadata["sub_scene_name"] = raw_metadata.get("sub_scene_code", "")
        converted_metadata["init_scene_text"] = raw_metadata.get("sub_scene_zh_dec", "")
        converted_metadata["english_init_scene_text"] = raw_metadata.get("sub_scene_en_dec", "")
        
        task_name = raw_metadata.get("task_group_name")
        if not task_name:
            task_name = raw_metadata.get("task_name", "")
        converted_metadata["task_name"] = task_name
        
        english_task_name = raw_metadata.get("task_group_code")
        if not english_task_name:
            english_task_name = raw_metadata.get("task_code", "")
        converted_metadata["english_task_name"] = english_task_name
        if isinstance(english_task_name, str) and "_" in english_task_name:
            english_task_name = english_task_name.replace("_", " ")
        converted_metadata["english_task_name"] = english_task_name
        
        converted_metadata["sn_code"] = raw_metadata.get("device_sn", "")

    # 默认值字段
    converted_metadata["data_type"] = "常规"
    converted_metadata["episode_status"] = "approved"
    converted_metadata["data_gen_mode"] = "real_machine"
    converted_metadata["sn_name"] = "乐聚机器人"

    print(f"Metadata 字段转换结果:")
    for key, value in converted_metadata.items():
        print(f"  {key}: '{value}'")

    # 使用转换后的 metadata
    metadata = converted_metadata

    # 获取时间信息
    rosbag_actual_start_time = None
    rosbag_actual_end_time = None
    rosbag_original_start_time = None
    rosbag_original_end_time = None

    # 如果传入了 main_time_line_timestamps，从时间戳计算总帧数
    # 否则使用传入的 total_frames 参数
    if main_time_line_timestamps is not None and len(main_time_line_timestamps) > 0:
        # 调试：打印原始时间戳
        print(f"原始时间戳前3个: {main_time_line_timestamps[:3]}")
        print(f"原始时间戳后3个: {main_time_line_timestamps[-3:]}")

        # 检查时间戳是否已经是秒格式还是纳秒格式
        if main_time_line_timestamps[0] > 1e12:  # 如果大于1e12，认为是纳秒格式
            timestamps_seconds = main_time_line_timestamps / 1e9
            print("时间戳格式：纳秒 -> 秒")
        else:
            timestamps_seconds = main_time_line_timestamps
            print("时间戳格式：已经是秒")

        rosbag_actual_start_time = timestamps_seconds[0]
        rosbag_actual_end_time = timestamps_seconds[-1]
        actual_total_frames = len(main_time_line_timestamps)
        print(f"从时间戳计算总帧数: {actual_total_frames}")
    else:
        # 使用传入的 total_frames 参数
        actual_total_frames = total_frames if total_frames is not None else 0
        print(f"使用传入的 total_frames: {actual_total_frames}")

    total_frames = actual_total_frames

    # 构造 action_config
    print(f"时间信息:")
    if rosbag_original_start_time and rosbag_original_end_time:
        print(
            f"  原始bag时间: {rosbag_original_start_time:.6f}s - {rosbag_original_end_time:.6f}s"
        )
    if rosbag_actual_start_time and rosbag_actual_end_time:
        print(
            f"  实际数据时间: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
        )
    print(f"  总帧数: {total_frames}")

    action_config = []

    # 根据格式选择数据源
    if is_new_format:
        # 新格式：从 marks 数组读取
        data_source = marks
    else:
        # 旧格式：从 moments 数组读取
        data_source = moment.get("moments", [])

    for m in data_source:
        if is_new_format:
            # 新格式：直接从 mark 对象读取
            mark_start = m.get("markStart", "")
            mark_end = m.get("markEnd", "")
            duration = m.get("duration", 0.0)  # 已经是数字，单位秒
            
            # 转换时间格式：从 "2026-01-06 09:41:20.781" 转为 ISO 格式
            try:
                # 解析 markStart 时间
                if mark_start:
                    # 尝试解析 "2026-01-06 09:41:20.781" 格式
                    if " " in mark_start:
                        dt_str, time_str = mark_start.split(" ", 1)
                        # 转换为 ISO 格式：2026-01-06T09:41:20.781+08:00
                        formatted_trigger_time = f"{dt_str}T{time_str}+08:00"
                    else:
                        formatted_trigger_time = mark_start
                else:
                    formatted_trigger_time = ""
            except Exception as e:
                print(f"[WARN] 解析 markStart 时间失败: {mark_start}, 错误: {e}")
                formatted_trigger_time = ""
            
            skill_atomic = m.get("skillAtomic", "")
            skill_detail = m.get("skillDetail", "")
            en_skill_detail = m.get("enSkillDetail", "")
            mark_type = m.get("markType", "step")
            
            # 判断是否为错误动作（retry 类型）
            is_mistake = (mark_type == "retry")
            
            print(f"处理动作数据（新格式）:")
            print(f"  skill_atomic: {skill_atomic}")
            print(f"  skill_detail: {skill_detail}")
            print(f"  en_skill_detail: {en_skill_detail}")
            print(f"  markStart: {mark_start}")
            print(f"  markEnd: {mark_end}")
            print(f"  duration: {duration}s")
            print(f"  markType: {mark_type} (is_mistake={is_mistake})")
        else:
            # 旧格式：从 customFieldValues 中提取数据
            custom_fields = m.get("customFieldValues", {})
            trigger_time = m.get("triggerTime", "")
            duration_str = m.get("duration", "0s")
            
            # 格式化时间戳：将 "Z" 替换为 "+00:00"
            formatted_trigger_time = (
                trigger_time.replace("Z", "+00:00") if trigger_time else ""
            )
            
            skill_atomic = custom_fields.get("skill_atomic_en", "")
            skill_detail = custom_fields.get("skill_detail", "")
            en_skill_detail = custom_fields.get("en_skill_detail", "")
            is_mistake = False  # 旧格式默认不是错误
            
            print(f"处理动作数据（旧格式）:")
            print(f"  skill_atomic_en: {skill_atomic}")
            print(f"  skill_detail: {skill_detail}")
            print(f"  en_skill_detail: {en_skill_detail}")
            print(f"  原始时间戳: {trigger_time}")
            print(f"  格式化时间戳: {formatted_trigger_time}")

        start_frame = None
        end_frame = None

        # 优先使用 fractional position（startPosition/endPosition）计算帧
        if is_new_format and total_frames is not None:
            # 新格式支持 fractional position (0.0-1.0)，优先使用它
            try:
                sp = m.get("startPosition")
                ep = m.get("endPosition")
                if sp is not None:
                    start_frame = int(float(sp) * total_frames)
                if ep is not None:
                    end_frame = int(float(ep) * total_frames)
                # clamp to valid range
                if start_frame is not None:
                    start_frame = max(0, min(start_frame, total_frames - 1))
                if end_frame is not None:
                    end_frame = max(start_frame if start_frame is not None else 0, min(end_frame, total_frames - 1))

                print(f"使用 fractional position 计算帧数: {start_frame} - {end_frame}")

            except Exception as e:
                print(f"[WARN] fractional position 计算失败: {e}")
                start_frame = None
                end_frame = None

        # 回退：如果没有 fractional position，使用绝对时间计算
        if (start_frame is None or end_frame is None) and (
            rosbag_actual_start_time is not None
            and rosbag_actual_end_time is not None
            and formatted_trigger_time
        ):
            try:
                if is_new_format:
                    # 新格式：使用 markStart 作为触发时间
                    # 解析 markStart 时间（格式：2026-01-06 09:41:20.781）
                    if mark_start and " " in mark_start:
                        dt_str, time_str = mark_start.split(" ", 1)
                        # 转换为 datetime 对象（假设是本地时间，+08:00）
                        trigger_datetime = datetime.datetime.fromisoformat(
                            f"{dt_str}T{time_str}+08:00"
                        )
                    else:
                        trigger_datetime = datetime.datetime.fromisoformat(
                            formatted_trigger_time
                        )
                    action_original_start_time = trigger_datetime.timestamp()
                    action_duration = float(duration)  # 已经是数字
                else:
                    # 旧格式：使用 triggerTime
                    trigger_datetime = datetime.datetime.fromisoformat(
                        formatted_trigger_time
                    )
                    action_original_start_time = trigger_datetime.timestamp()
                    # 解析持续时间
                    action_duration = 0
                    if duration_str.endswith("s"):
                        action_duration = float(duration_str[:-1])

                # 计算帧数
                start_frame, end_frame = calculate_action_frames(
                    rosbag_actual_start_time=rosbag_actual_start_time,
                    rosbag_actual_end_time=rosbag_actual_end_time,
                    rosbag_original_start_time=rosbag_original_start_time,
                    rosbag_original_end_time=rosbag_original_end_time,
                    action_original_start_time=action_original_start_time,
                    action_duration=action_duration,
                    frame_rate=frequency,
                    total_frames=total_frames,
                )

                print(f"动作: {skill_detail}")
                print(f"  动作时间: {trigger_datetime.isoformat()}")
                print(f"  原始开始时间: {action_original_start_time:.6f}s")
                print(
                    f"  原始结束时间: {action_original_start_time + action_duration:.6f}s"
                )
                print(f"  持续时间: {action_duration:.3f}s")
                print(f"  计算得到帧数: {start_frame} - {end_frame}")

                # 更详细的调试信息
                if start_frame is None or end_frame is None:
                    print(f"  调试信息:")
                    print(
                        f"    实际数据范围: {rosbag_actual_start_time:.6f}s - {rosbag_actual_end_time:.6f}s"
                    )
                    print(
                        f"    动作时间范围: {action_original_start_time:.6f}s - {action_original_start_time + action_duration:.6f}s"
                    )
                    print(
                        f"    动作是否在数据范围内: {action_original_start_time >= rosbag_actual_start_time and action_original_start_time + action_duration <= rosbag_actual_end_time}"
                    )
                    print(
                        f"    动作开始是否在数据范围后: {action_original_start_time > rosbag_actual_end_time}"
                    )
                    print(
                        f"    动作结束是否在数据范围前: {action_original_start_time + action_duration < rosbag_actual_start_time}"
                    )

                # 验证计算结果
                if start_frame is not None and end_frame is not None:
                    actual_start_time = rosbag_actual_start_time + (
                        start_frame / total_frames
                    ) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    actual_end_time = rosbag_actual_start_time + (
                        end_frame / total_frames
                    ) * (rosbag_actual_end_time - rosbag_actual_start_time)
                    print(f"  验证-实际开始时间: {actual_start_time:.6f}s")
                    print(f"  验证-实际结束时间: {actual_end_time:.6f}s")

                print("-" * 50)

            except Exception as e:
                print(f"计算帧数时出错: {e}")
                import traceback

                traceback.print_exc()

        # 构造新的 action 对象
        action = {
            "start_frame": start_frame,
            "end_frame": end_frame,
            "timestamp_utc": formatted_trigger_time,
            "is_mistake": is_mistake,
            "skill": skill_atomic,
            "action_text": skill_detail,
            "english_action_text": en_skill_detail,
        }
        action_config.append(action)

    # 按照 timestamp_utc 排序
    action_config = sorted(
        action_config, key=lambda x: x["timestamp_utc"] if x["timestamp_utc"] else ""
    )

    # 规范化帧范围（确保区间连续且覆盖整个 episode）
    if action_config and total_frames is not None:
        # 1. 第一个动作从 0 开始（强制规范化）
        first = action_config[0]

        # 强制第一个动作的 start_frame 为 0
        first["start_frame"] = 0

        # 如果没有 end_frame，设置为 start_frame
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

    # 构造新json，episode_id放在最前
    new_json = OrderedDict()
    new_json["episode_id"] = uuid

    # 使用转换后的 metadata
    for k, v in metadata.items():
        new_json[k] = v

    if "label_info" not in new_json:
        new_json["label_info"] = {}
    new_json["label_info"]["action_config"] = action_config
    if "key_frame" not in new_json["label_info"]:
        new_json["label_info"]["key_frame"] = []

    # 添加 total_frames 字段
    new_json["total_frames"] = total_frames

    # 保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_json, f, ensure_ascii=False, indent=4)
    print(f"已保存到 {output_path} (总帧数: {total_frames})")


def load_metadata(path: str) -> dict:
    """读取 metadata.json 并返回字典，如果失败返回空字典。

    Args:
        path: metadata.json 文件路径

    Returns:
        Python 字典或空字典。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[load_metadata] 无法读取 {path}: {e}")
        return {}


def extract_robot_model(metadata: dict) -> dict:
    """从 metadata 中提取可用于覆盖配置的字段。

    Supported keys include:
        - robot_model or model
        - urdf_path
        - source_topics
        - camera_topic_specs
        - robot_profile

    Returns a dict containing whichever keys were present.
    """
    overrides = {}
    for key in ("robot_model", "model", "urdf_path", "source_topics", "camera_topic_specs", "robot_profile"):
        if key in metadata:
            overrides[key] = metadata[key]
    return overrides


def get_sn_code(metadata: dict) -> Optional[str]:
    """返回 deviceSn 或 device_sn 字段，用于判断摄像头翻转等。

    Args:
        metadata: 元数据字典

    Returns:
        字符串或 None。
    """
    return metadata.get("deviceSn") or metadata.get("device_sn")


def get_time_range_from_metadata(metadata: dict) -> Optional[tuple]:
    """从 metadata 字典中提取时间范围（start/end positions）。

    如果是新格式（包含 marks 数组），会返回 min(startPosition) 和 max(endPosition)。
    否则返回 None, None.
    """
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


def merge_batch_metadata(batch_dirs: List[str], output_dir: str, total_frames: int) -> dict:
    """将多个 batch 的 metadata 合并成一个 episode 级别的 metadata.json。

    Args:
        batch_dirs: 各 batch 目录路径列表，每个目录内应包含 metadata.json
        output_dir: 合并后 metadata 存放目录（会创建该目录）
        total_frames: 最终总帧数，用于修正最后一条动作的 end_frame

    Returns:
        合并后的 metadata 字典。
    """
    merged = {"total_frames": total_frames, "label_info": {"action_config": []}}
    current_offset = 0

    for bdir in batch_dirs:
        mp = os.path.join(bdir, "metadata.json")
        if not os.path.isfile(mp):
            print(f"[merge_batch] 缺少 batch metadata: {mp}")
            continue
        md = load_metadata(mp)
        actions = md.get("label_info", {}).get("action_config", [])

        # 计算本 batch 的帧数
        frames = None
        if isinstance(md.get("total_frames"), int):
            frames = md.get("total_frames")
        else:
            # 回退：读取 parquet 文件计行数
            frames = 0
            import glob
            try:
                import pyarrow.parquet as pq
            except ImportError:
                pq = None
            files = glob.glob(os.path.join(bdir, "**", "*.parquet"), recursive=True)
            for f in files:
                try:
                    if pq is not None:
                        tbl = pq.ParquetFile(f)
                        frames += tbl.metadata.num_rows
                    else:
                        # without pyarrow, attempt pandas
                        import pandas as pd
                        frames += len(pd.read_parquet(f, columns=["timestamp"]))
                except Exception:
                    pass
        if frames is None:
            frames = 0

        # 合并 actions 并添加偏移
        for act in actions:
            if act.get("start_frame") is not None:
                act["start_frame"] = act["start_frame"] + current_offset
            if act.get("end_frame") is not None:
                act["end_frame"] = act["end_frame"] + current_offset
            merged["label_info"]["action_config"].append(act)

        current_offset += frames

    # 排序并去重
    merged_actions = merged["label_info"]["action_config"]
    merged_actions = sorted(
        merged_actions,
        key=lambda x: (x.get("timestamp_utc", ""), x.get("start_frame", 0)),
    )
    unique = []
    seen = set()
    for act in merged_actions:
        key = (act.get("skill"), act.get("timestamp_utc"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(act)
    merged_actions = unique

    # 修正首末帧并规范化为连续区间
    if merged_actions:
        # 1. 第一个动作从 0 开始
        first = merged_actions[0]
        if first.get("start_frame") is None:
            first["start_frame"] = 0
        else:
            first["start_frame"] = max(0, min(int(first["start_frame"]), total_frames - 1))

        if first.get("end_frame") is None:
            first["end_frame"] = first["start_frame"]
        else:
            first["end_frame"] = max(first["start_frame"], min(int(first["end_frame"]), total_frames - 1))

        # 2. 后续动作：起点衔接上一个动作的 end_frame
        for i in range(1, len(merged_actions)):
            prev = merged_actions[i - 1]
            cur = merged_actions[i]

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
        last = merged_actions[-1]
        last["end_frame"] = total_frames - 1

    merged["label_info"]["action_config"] = merged_actions

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "metadata.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=4)
        print(f"[merge_batch] 合并 metadata 写入 {out_path}")
    except Exception as e:
        print(f"[merge_batch] 写出失败: {e}")

    return merged


def get_time_range_from_moments(moments_json_path, metadata_json_path=None):
    """
    从 moments.json 或 metadata.json（新格式）文件中读取时间范围
    支持两种格式：
    1. 旧格式：从 moments.json 的 moments 数组中读取 start_position/end_position
    2. 新格式：从 metadata.json 的 marks 数组中读取 startPosition/endPosition

    Args:
        moments_json_path: moments.json 文件路径（旧格式）
        metadata_json_path: metadata.json 文件路径（新格式，可选）

    Returns:
        tuple: (start_time, end_time) 或 (None, None) 如果失败
    """
    # 优先尝试从新格式的 metadata.json 读取
    if metadata_json_path and os.path.exists(metadata_json_path):
        try:
            with open(metadata_json_path, "r", encoding="utf-8") as f:
                metadata_data = json.load(f)
            
            # 检查是否为新格式（包含 marks 字段）
            if "marks" in metadata_data and isinstance(metadata_data.get("marks"), list):
                marks = metadata_data.get("marks", [])
                if not marks:
                    print(f"[MOMENTS] metadata.json中未找到marks数据")
                else:
                    start_positions = []
                    end_positions = []
                    
                    for mark in marks:
                        start_pos = mark.get("startPosition")
                        end_pos = mark.get("endPosition")
                        
                        if start_pos is not None:
                            try:
                                start_positions.append(float(start_pos))
                            except (ValueError, TypeError):
                                print(f"[MOMENTS] 无效的startPosition值: {start_pos}")
                                pass
                        
                        if end_pos is not None:
                            try:
                                end_positions.append(float(end_pos))
                            except (ValueError, TypeError):
                                print(f"[MOMENTS] 无效的endPosition值: {end_pos}")
                                pass
                    
                    if start_positions and end_positions:
                        moments_start_time = min(start_positions)
                        moments_end_time = max(end_positions)
                        
                        print(
                            f"[MOMENTS] 从metadata.json（新格式）获取时间范围: {moments_start_time} - {moments_end_time}"
                        )
                        print(
                            f"[MOMENTS] 找到 {len(start_positions)} 个startPosition, {len(end_positions)} 个endPosition"
                        )
                        
                        return moments_start_time, moments_end_time
                    else:
                        print(f"[MOMENTS] metadata.json中未找到有效的时间位置信息")
        except Exception as e:
            print(f"[MOMENTS] 读取metadata.json时出错: {e}")
    
    # 回退到旧格式：从 moments.json 读取
    if not moments_json_path or not os.path.exists(moments_json_path):
        return None, None

    try:
        with open(moments_json_path, "r", encoding="utf-8") as f:
            moments_data = json.load(f)

        moments = moments_data.get("moments", [])
        if not moments:
            print(f"[MOMENTS] moments.json中未找到moments数据")
            return None, None

        start_positions = []
        end_positions = []

        for moment in moments:
            custom_fields = moment.get("customFieldValues", {})
            start_pos = custom_fields.get("start_position")
            end_pos = custom_fields.get("end_position")

            if start_pos is not None:
                try:
                    start_positions.append(float(start_pos))
                except (ValueError, TypeError):
                    print(f"[MOMENTS] 无效的start_position值: {start_pos}")
                    pass

            if end_pos is not None:
                try:
                    end_positions.append(float(end_pos))
                except (ValueError, TypeError):
                    print(f"[MOMENTS] 无效的end_position值: {end_pos}")
                    pass

        # 使用最早的start_position和最晚的end_position
        if start_positions and end_positions:
            moments_start_time = min(start_positions)
            moments_end_time = max(end_positions)

            print(
                f"[MOMENTS] 从moments.json获取时间范围: {moments_start_time} - {moments_end_time}"
            )
            print(
                f"[MOMENTS] 找到 {len(start_positions)} 个start_position, {len(end_positions)} 个end_position"
            )

            return moments_start_time, moments_end_time
        else:
            print(f"[MOMENTS] moments.json中未找到有效的时间位置信息")
            return None, None

    except Exception as e:
        print(f"[MOMENTS] 读取moments.json时出错: {e}")
        return None, None
