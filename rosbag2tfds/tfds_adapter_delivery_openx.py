#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TFDS adapter for the Open-X delivery dataset."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


MAX_AUX_STREAMS = 8
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

def _build_dataset_docs(meta: Dict) -> Tuple[str, str]:
    """Generate dataset description/doc strings from metadata."""
    if not meta:
        default_desc = (
            "Delivery robot demonstrations captured as ROS1 bags and exported to the "
            "Open-X RLDS schema. Each episode corresponds to a single bag recording "
            "aligned to the camera timeline."
        )
        default_doc = (
            "Key fields\n==========\n\nObservation\n-----------\n"
            "- image_primary/image: Head RGB stream (/cam_h/color/image_raw/compressed).\n"
            "- image_aux_k/image: Auxiliary RGB/Depth streams.\n"
            "- state/joint_* & state/eef_*: Proprioception from /sensors_data_raw and hand topics.\n"
            "- timestamp: Nanoseconds aligned to primary camera.\n"
            "- subtask_language_instruction: Step-level hint from sidecar marks.\n\n"
            "Action\n------\n"
            "- agent/joint_*: /joint_cmd.{joint_q,v,tau} commands.\n"
            "- agent/eef_*: /leju_claw_command or dexhand command topics.\n\n"
            "Episode metadata\n----------------\n"
            "- camera intrinsics/extrinsics serialized to metadata JSON.\n"
            "- sidecar_json: Raw prompts, marks, and annotations."
        )
        return default_desc, default_doc

    dataset_name = meta.get("dataset_name", "delivery_openx")
    bag_path = meta.get("bag_path") or meta.get("episode_id") or "N/A"
    num_steps = meta.get("num_steps")
    timeline = meta.get("timeline", "camera")
    eef_type = meta.get("eef_type", "unknown")
    rgb_topics = meta.get("rgb_topics", [])
    depth_topics = meta.get("depth_topics", [])
    marks = meta.get("marks_by_step") or []

    desc_lines = [
        f"{dataset_name} episode converted from bag: {bag_path}",
        f"Aligned timeline: {timeline}, steps: {num_steps if num_steps is not None else 'unknown'}.",
        f"End-effector type: {eef_type}, joint_state topic: {meta.get('sensors_data_raw_topic', 'N/A')}.",
        "Camera streams:",
    ]
    if rgb_topics:
        desc_lines += [f"  • RGB: {topic}" for topic in rgb_topics]
    if depth_topics:
        desc_lines += [f"  • Depth: {topic}" for topic in depth_topics]
    if not (rgb_topics or depth_topics):
        desc_lines.append("  • (no camera topics recorded)")
    if marks:
        desc_lines.append(f"Sidecar marks: {len(marks)} segment(s) mapped to subtask_language_instruction.")
    desc_lines.append("Global instruction stored per step in observation/natural_language_instruction.")

    shard_list = meta.get("shards", [])
    shard_preview = ", ".join(shard_list[:2]) + (" ..." if len(shard_list) > 2 else "")
    doc_lines = [
        "Observation",
        "-----------",
        "- image_primary/image: main camera decoded frame (tfds.features.Image).",
        "- image_aux_k/image: auxiliary RGB/Depth streams (see observation/image_aux_k/{camera,topic}).",
        "- state/joint_position|velocity|torque: /sensors_data_raw.joint_data.{joint_q,v,tau}.",
        "- state/eef_*: Hand/claw state padded to 12 dims.",
        "- state/tcp_*: FK-based TCP pose (requires URDF during conversion).",
        "- timestamp: int64 ns aligned to the primary camera timeline.",
        "- camera_extrinsics_json: per-step extrinsics JSON (TF-derived or URDF fallback).",
        "- subtask_language_instruction: step-level text from marks_by_step.",
        "",
        "Action",
        "------",
        "- agent/joint_*: /joint_cmd outputs.",
        "- agent/eef_*: /leju_claw_command or dexhand command topic.",
        "- world_vector / rotation_delta: TCP deltas (left arm mirrored to main action fields).",
        "",
        "Episode metadata",
        "----------------",
        "- camera_intrinsics_json / camera_extrinsics_json: per-camera parameters (URDF or TF fallback).",
        "- camera_info: ROS CameraInfo snapshot for each camera slot.",
        "- sidecar_json & marks_by_step: language annotations and timeline segments.",
        f"- shards: {shard_preview if shard_preview else '(not recorded)'}",
    ]

    return "\n".join(desc_lines), "\n".join(doc_lines)


def _image_branch(
    name: str,
    *,
    dtype: tf.dtypes.DType,
    channels: int,
) -> tfds.features.FeaturesDict:
    """Construct a TFDS image branch with explicit dtype/channel config."""
    shape = (None, None, channels)
    return tfds.features.FeaturesDict({
        "image": tfds.features.Image(
            encoding_format=None,
            dtype=dtype,
            shape=shape,
            doc=f"Decoded frame for {name} stored as tfds.features.Image ({dtype.name}, {channels} channel(s)).",
        ),
        "bit_depth": tfds.features.Tensor(
            shape=(),
            dtype=tf.int32,
            doc=f"Bit depth inferred from the encoded data for {name}.",
        ),
        "format": tfds.features.Text(
            doc=f"Original encoding format reported by ROS for {name} (jpeg/png/etc)."
        ),
        "camera": tfds.features.Text(
            doc=f"Logical camera label (head/left/right/etc.) associated with {name}."
        ),
        "topic": tfds.features.Text(
            doc=f"ROS topic captured into {name}."
        ),
        "stream_type": tfds.features.Text(
            doc=f"Modality for {name}, e.g. 'rgb' or 'depth'."
        ),
    })


def _strip_png_padding(data: bytes) -> Tuple[bytes, int]:
    """Remove any leading padding before the PNG signature."""
    if not data:
        return data, 0
    idx = data.find(PNG_SIGNATURE)
    if idx > 0:
        return data[idx:], idx
    return data, 0


def _infer_png_bit_depth(data: bytes) -> int:
    """Return the bit depth declared in the PNG IHDR chunk."""
    if not data or len(data) < 24:
        return 8
    if data[:8] != PNG_SIGNATURE:
        return 8
    ihdr_start = 8
    if len(data) < ihdr_start + 25:
        return 8
    ihdr_data = data[ihdr_start + 8:ihdr_start + 21]
    if len(ihdr_data) != 13:
        return 8
    return int(ihdr_data[8])


class DeliveryOpenx(tfds.core.GeneratorBasedBuilder):
    """TFDS builder for delivery_openx RLDS dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release of delivery_openx dataset."
    }

    def __init__(self, **kwargs):
        self._source_dir = kwargs.pop("source_dir", None)
        self._meta: Dict = {}
        self._shards: List[str] = []
        self._description: str = ""
        self._doc: str = ""
        self._load_source_metadata()
        if not self._description:
            self._description, self._doc = _build_dataset_docs(self._meta)
        super().__init__(**kwargs)
        self._update_metadata_cache()

    def _load_source_metadata(self):
        if not self._source_dir:
            return
        meta_path = Path(self._source_dir) / "rlds_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self._meta = json.load(f)
                    self._description, self._doc = _build_dataset_docs(self._meta)
            except Exception:
                self._meta = {}
                self._description, self._doc = _build_dataset_docs({})
        # Don't call _update_metadata_cache() here - it will be called after super().__init__()

    def _update_metadata_cache(self):
        if not hasattr(self, "info") or self.info is None:
            return
        meta = self._meta or {}
        self.info.metadata["camera_info"] = meta.get("camera_info", {})
        self.info.metadata["marks_by_step"] = meta.get("marks_by_step", [])
        self.info.metadata["sidecar"] = meta.get("sidecar_meta", {})

    def _reshard_tfds_split(
        self,
        split_dir: Path,
        dataset_name: str,
        split: str,
        desired_num_shards: int,
    ) -> List[Path]:
        import tensorflow as tf

        current_shards = sorted(split_dir.glob("*.tfrecord*"))
        if len(current_shards) == desired_num_shards:
            return current_shards

        if desired_num_shards <= 0:
            raise ValueError("desired_num_shards must be positive when re-sharding TFDS output")

        print(
            f"   ⚠️ TFDS 实际生成 {len(current_shards)} 个分片，与期望 {desired_num_shards} 不符，开始重新分片"
        )
        tmp_dir = split_dir / "__tmp_reshard"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        writers: List[tf.io.TFRecordWriter] = []
        tmp_paths: List[Path] = []
        for idx in range(desired_num_shards):
            name = f"{dataset_name}-{split}.tfrecord-{idx:05d}-of-{desired_num_shards:05d}"
            tmp_path = tmp_dir / name
            writers.append(tf.io.TFRecordWriter(str(tmp_path)))
            tmp_paths.append(tmp_path)

        shard_index = 0
        for src in current_shards:
            for raw in tf.compat.v1.io.tf_record_iterator(str(src)):
                writers[shard_index].write(raw)
                shard_index = (shard_index + 1) % desired_num_shards

        for writer in writers:
            writer.close()

        for src in current_shards:
            src.unlink(missing_ok=True)

        new_paths: List[Path] = []
        for tmp_path in tmp_paths:
            dest = split_dir / tmp_path.name
            tmp_path.replace(dest)
            new_paths.append(dest)

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"   ✓ 重新分片完成，共 {len(new_paths)} 个 TFDS 分片")
        return new_paths

    def export_static_tfds(
        self,
        target_root: Union[str, Path],
        split: str = "train",
        *,
        clean_target: bool = True,
        desired_num_shards: Optional[int] = None,
    ) -> Dict[str, object]:
        """Copy generated TFDS artifacts into a target directory."""

        if not target_root:
            raise ValueError("target_root must be provided when exporting TFDS artifacts")

        dataset_name = getattr(self, "name", None) or (
            self.info.full_name.split("/")[0] if getattr(self, "info", None) else "delivery_openx"
        )
        version_str = str(self.VERSION)
        generated_dir = Path(self.data_dir)
        if not generated_dir.exists():
            raise FileNotFoundError(f"Builder data_dir does not exist: {generated_dir}")

        source_split_dir = generated_dir / split
        shard_sources: List[Path]
        if source_split_dir.exists():
            shard_sources = sorted(source_split_dir.glob("*.tfrecord*"))
        else:
            pattern = f"{dataset_name}-{split}.tfrecord-*"
            shard_sources = sorted(generated_dir.glob(pattern))
            source_split_dir = generated_dir
        if not shard_sources:
            raise FileNotFoundError(
                f"TFDS split directory missing shard files. looked in {source_split_dir}"
            )

        target_root_path = Path(target_root).expanduser().resolve()
        target_version_dir = target_root_path / dataset_name / version_str
        target_split_dir = target_version_dir / split
        target_version_dir.mkdir(parents=True, exist_ok=True)

        if clean_target and target_split_dir.exists():
            shutil.rmtree(target_split_dir)
        target_split_dir.mkdir(parents=True, exist_ok=True)

        copied_shards: List[Path] = []
        for shard_path in shard_sources:
            if not shard_path.is_file():
                continue
            dest_path = target_split_dir / shard_path.name
            shutil.copy2(shard_path, dest_path)
            copied_shards.append(dest_path)

        if not copied_shards:
            raise RuntimeError(
                f"No TFDS shards were copied from {source_split_dir}. Ensure download_and_prepare() completed successfully."
            )

        if desired_num_shards and desired_num_shards > 0:
            if len(copied_shards) != desired_num_shards:
                copied_shards = self._reshard_tfds_split(
                    target_split_dir,
                    dataset_name,
                    split,
                    desired_num_shards,
                )

        tfds_shards = [str(path.relative_to(target_root_path)) for path in sorted(copied_shards)]

        metadata_files: List[str] = []
        for json_name in ("dataset_info.json", "features.json", "metadata.json"):
            src = generated_dir / json_name
            if not src.exists():
                raise FileNotFoundError(f"Missing TFDS metadata file: {src}")
            dst = target_version_dir / json_name
            shutil.copy2(src, dst)
            metadata_files.append(str(dst.relative_to(target_root_path)))

        return {
            "dataset_name": dataset_name,
            "version": version_str,
            "tfds_dir": str(target_version_dir),
            "split": split,
            "split_dir": str(target_split_dir),
            "tfds_shards": tfds_shards,
            "metadata_files": metadata_files,
        }

    def _camera_slot_specs(self) -> Tuple[Dict[str, str], List[Dict[str, str]]]:
        """Return primary and auxiliary camera stream specs inferred from metadata."""
        main_topic = self._meta.get("main_rgb_topic") or ""
        rgb_topics = self._meta.get("rgb_topics", [])
        depth_topics = self._meta.get("depth_topics", [])
        specs: List[Dict[str, str]] = []
        for topic in rgb_topics:
            role = "primary" if topic == main_topic else "aux"
            specs.append({"topic": topic, "kind": "rgb", "role": role})
        for topic in depth_topics:
            specs.append({"topic": topic, "kind": "depth", "role": "aux"})
        primary = next((s for s in specs if s["role"] == "primary"), {"topic": main_topic, "kind": "rgb", "role": "primary"})
        aux = [s for s in specs if s["role"] == "aux"]
        return primary, aux

    def _info(self) -> tfds.core.DatasetInfo:
        # Get joint names from metadata to generate dynamic doc strings
        joint_names = self._meta.get("joint_names", [])
        # Use actual tensor shape (28) instead of joint_names length
        # The tensor shape is fixed at 28, even if joint_names has more entries
        num_joints = 28  # Fixed tensor shape
        
        # Get EEF info from metadata
        eef_type = self._meta.get("eef_type", "unknown")
        eef_dim = self._meta.get("eef_dim", 12)
        
        # Generate joint structure description based on actual tensor shape
        # Note: Only first 28 joints from joint_names are used in the tensor
        if joint_names and len(joint_names) >= num_joints:
            # Use first num_joints joints for description
            used_joint_names = joint_names[:num_joints]
            # Try to identify structure (e.g., left/right arms, legs)
            left_joints = [name for name in used_joint_names if 'left' in name.lower() or name.startswith('l_') or '_l' in name.lower()]
            right_joints = [name for name in used_joint_names if 'right' in name.lower() or name.startswith('r_') or '_r' in name.lower()]
            leg_joints = [name for name in used_joint_names if 'leg' in name.lower()]
            arm_joints = [name for name in used_joint_names if 'arm' in name.lower() or 'zarm' in name.lower()]
            
            # Build description based on actual structure
            parts = []
            if leg_joints:
                parts.append(f"{len(leg_joints)} leg joints")
            if arm_joints:
                parts.append(f"{len(arm_joints)} arm joints")
            if left_joints and not right_joints:
                parts.append(f"{len(left_joints)} left-side hand joints")
            elif right_joints and not left_joints:
                parts.append(f"{len(right_joints)} right-side hand joints")
            elif left_joints and right_joints:
                parts.append(f"{len(left_joints)} left-side and {len(right_joints)} right-side hand joints")
            
            structure_summary = ", ".join(parts) if parts else "mixed joints"
            
            joint_structure_desc = (
                f"Structure: {num_joints}-DoF joint values. "
                f"Order: {structure_summary}. "
                f"First few joints: {', '.join(used_joint_names[:5])}... "
                f"(see episode_metadata/joint_names_json for full list of {len(joint_names)} total joints, "
                f"but only first {num_joints} are used in this tensor)."
            )
        elif joint_names:
            # Fewer joint names than tensor size
            joint_structure_desc = (
                f"Structure: {num_joints}-DoF joint values. "
                f"Available joint names: {', '.join(joint_names[:5])}... "
                f"(see episode_metadata/joint_names_json for full list of {len(joint_names)} joints)."
            )
        else:
            joint_structure_desc = (
                f"Structure: {num_joints}-DoF joint values. "
                f"Joint order: Not provided. See episode_metadata/joint_names_json if available."
            )
        
        # Generate EEF structure description from metadata
        if eef_type == "dexhand":
            eef_obs_desc = (
                f"End-effector state from /dexhand/state. "
                f"Structure: 12-D hand joint positions/angles (observation) or velocities (rad/s) or torques/efforts (N·m). "
                f"All 12 dimensions are used (no padding). "
                f"Source topic: /dexhand/state."
            )
            eef_action_desc = (
                f"End-effector command from /control_robot_hand_position. "
                f"Structure: 12-D hand joint position commands. "
                f"Velocity command is always zeros (not actively commanded for dexhand). "
                f"Source topic: /control_robot_hand_position."
            )
        elif eef_type == "leju_claw":
            eef_obs_desc = (
                f"End-effector state from /leju_claw_state. "
                f"Structure: {eef_dim}-D claw state (position/velocity/effort) padded to 12 dims. "
                f"First {eef_dim} dims contain actual values, remaining {12 - eef_dim} dims are zeros. "
                f"Source topic: /leju_claw_state."
            )
            eef_action_desc = (
                f"End-effector command from /leju_claw_command. "
                f"Structure: {eef_dim}-D claw position/velocity command padded to 12 dims. "
                f"First {eef_dim} dims contain actual values, remaining {12 - eef_dim} dims are zeros. "
                f"Source topic: /leju_claw_command."
            )
        else:
            eef_obs_desc = (
                f"End-effector state. Structure: Padded to 12 dims for Open-X compatibility. "
                f"Actual dimension: {eef_dim} (see episode_metadata/eef_dim)."
            )
            eef_action_desc = (
                f"End-effector command. Structure: Padded to 12 dims for Open-X compatibility. "
                f"Actual dimension: {eef_dim} (see episode_metadata/eef_dim)."
            )
        
        primary_spec, aux_specs = self._camera_slot_specs()

        def _branch_config(kind: str) -> Tuple[tf.dtypes.DType, int]:
            if (kind or "").lower() == "depth":
                return tf.uint16, 1
            return tf.uint8, 3

        prim_dtype, prim_channels = _branch_config(primary_spec.get("kind", "rgb"))
        num_aux = int(self._meta.get("num_aux_streams", len(aux_specs)))
        num_aux = max(0, min(num_aux, len(aux_specs), MAX_AUX_STREAMS))
        observation_images = {
            "image_primary": _image_branch(
                f"image_primary ({primary_spec.get('topic', self._meta.get('main_rgb_topic', 'unknown'))})",
                dtype=prim_dtype,
                channels=prim_channels,
            ),
        }
        for i in range(1, num_aux + 1):
            spec = aux_specs[i - 1]
            aux_dtype, aux_channels = _branch_config(spec.get("kind", "rgb"))
            observation_images[f"image_aux_{i}"] = _image_branch(
                f"image_aux_{i} ({spec.get('topic', 'auxiliary camera stream')})",
                dtype=aux_dtype,
                channels=aux_channels,
            )

        observation = tfds.features.FeaturesDict({
            **observation_images,
            "state": tfds.features.FeaturesDict({
                "joint_position": tfds.features.Tensor(
                    shape=(28,), dtype=tf.float32,
                    doc=f"Joint positions in radians. Source: /sensors_data_raw.joint_data.joint_q. "
                        f"{joint_structure_desc}"
                ),
                "joint_velocity": tfds.features.Tensor(
                    shape=(28,), dtype=tf.float32,
                    doc=f"Joint velocities in rad/s. Source: /sensors_data_raw.joint_data.joint_v. "
                        f"{joint_structure_desc}"
                ),
                "joint_torque": tfds.features.Tensor(
                    shape=(28,), dtype=tf.float32,
                    doc=f"Joint torques in N·m. Source: /sensors_data_raw.joint_data.joint_torque. "
                        f"{joint_structure_desc} "
                        f"Measured joint torques from robot sensors."
                ),
                "eef_position": tfds.features.Tensor(
                    shape=(12,), dtype=tf.float32,
                    doc=f"End-effector position/pose state. {eef_obs_desc} "
                        f"Units: radians for joint angles, meters for positions (if applicable)."
                ),
                "eef_velocity": tfds.features.Tensor(
                    shape=(12,), dtype=tf.float32,
                    doc=f"End-effector velocity. {eef_obs_desc} "
                        f"Units: rad/s for joint velocities."
                ),
                "eef_effort": tfds.features.Tensor(
                    shape=(12,), dtype=tf.float32,
                    doc=f"End-effector effort/torque. {eef_obs_desc} "
                        f"Units: N·m (Newton-meters) for torques."
                ),
                "tcp_position_left": tfds.features.Tensor(
                    shape=(3,), dtype=tf.float32,
                    doc="Left arm TCP (Tool Center Point) position [x, y, z] in meters relative to base_link. "
                        "Computed using Forward Kinematics from joint positions via Drake library. "
                        "This is the actual end-effector position, not hand joint angles."
                ),
                "tcp_position_right": tfds.features.Tensor(
                    shape=(3,), dtype=tf.float32,
                    doc="Right arm TCP (Tool Center Point) position [x, y, z] in meters relative to base_link. "
                        "Computed using Forward Kinematics from joint positions via Drake library. "
                        "This is the actual end-effector position, not hand joint angles."
                ),
                "tcp_orientation_left": tfds.features.Tensor(
                    shape=(4,), dtype=tf.float32,
                    doc="Left arm TCP orientation as quaternion [x, y, z, w] relative to base_link. "
                        "Computed using Forward Kinematics from joint positions via Drake library. "
                        "Format: ROS standard quaternion [x, y, z, w]."
                ),
                "tcp_orientation_right": tfds.features.Tensor(
                    shape=(4,), dtype=tf.float32,
                    doc="Right arm TCP orientation as quaternion [x, y, z, w] relative to base_link. "
                        "Computed using Forward Kinematics from joint positions via Drake library. "
                        "Format: ROS standard quaternion [x, y, z, w]."
                ),
                "vr_tcp_pose": tfds.features.Tensor(
                    shape=(14,), dtype=tf.float32,
                    doc="VR end-effector pose from /ik_fk_result/eef_pose topic. "
                        "Format: 14 floats = left_pos[3] + left_quat[4] + right_pos[3] + right_quat[4]. "
                        "Position in meters, quaternion in [x, y, z, w] format. "
                        "Source: kuavo_msgs/twoArmHandPose message. "
                        "Used for VR-based imitation learning."
                ),
            }),
            "timestamp": tfds.features.Tensor(
                shape=(), dtype=tf.int64,
                doc="Timeline-aligned ROS timestamp in nanoseconds. "
                    "Source: ROS message header stamp, aligned to the primary camera frame timeline. "
                    "Used for temporal synchronization across all sensors. "
                    "Units: nanoseconds (ROS time format)."
            ),
            "natural_language_instruction": tfds.features.Text(
                doc="Episode-level natural language instruction describing the overall task goal. "
                    "This instruction remains constant across all steps within an episode. "
                    "Source: sidecar JSON 'globalInstruction' field (or 'NULL' if not present). "
                    "Used by Open-X models (RT-1, RT-1-X) for training and inference. "
                    "Example: 'pick and place packages', 'deliver items to storage area'."
            ),
            "natural_language_instruction_1": tfds.features.Text(
                doc="First alternative natural language instruction variant for data augmentation. "
                    "Source: sidecar JSON 'globalInstructionVariants[0]' (from taskRemark, split by ';'). "
                    "Empty string if not provided."
            ),
            "natural_language_instruction_2": tfds.features.Text(
                doc="Second alternative natural language instruction variant for data augmentation. "
                    "Source: sidecar JSON 'globalInstructionVariants[1]' (from taskRemark, split by ';'). "
                    "Empty string if not provided."
            ),
            "natural_language_instruction_3": tfds.features.Text(
                doc="Third alternative natural language instruction variant for data augmentation. "
                    "Source: sidecar JSON 'globalInstructionVariants[2]' (from taskRemark, split by ';'). "
                    "Empty string if not provided."
            ),
            "subtask_language_instruction": tfds.features.Text(
                doc="Step-level language instruction describing the current sub-task. "
                    "Source: sidecar JSON marks (enDesc/enSkillDetail fields). "
                    "Dynamically changes per step based on the active mark. "
                    "Example: 'pick package from conveyor belt', 'place package on storage machine'. "
                    "Empty string if no mark is active for the step. "
                    "This provides fine-grained task description in addition to the episode-level natural_language_instruction."
            ),
            "camera_extrinsics_json": tfds.features.Text(
                doc="Per-step camera extrinsics (JSON) derived from TF (if available) or URDF fallback. "
                    "Contains parent_link/child_link/xyz/rpy/transform_matrix for each camera."
            ),
        })

        action = tfds.features.FeaturesDict({
            # Original joint/eef commands (for low-level control)
            "agent": tfds.features.FeaturesDict({
                "joint_position": tfds.features.Tensor(
                    shape=(28,), dtype=tf.float32,
                    doc=f"Commanded joint positions in radians. Source: /joint_cmd.joint_q. "
                        f"Same order as observation/state/joint_position. {joint_structure_desc}"
                ),
                "joint_velocity": tfds.features.Tensor(
                    shape=(28,), dtype=tf.float32,
                    doc=f"Commanded joint velocities in rad/s. Source: /joint_cmd.joint_v. "
                        f"Same order as observation/state/joint_velocity. {joint_structure_desc}"
                ),
                "joint_torque": tfds.features.Tensor(
                    shape=(28,), dtype=tf.float32,
                    doc=f"Commanded joint torques in N·m. Source: /joint_cmd.tau. "
                        f"Feedforward torque commands sent to robot actuators. "
                        f"Same order as observation/state/joint_torque."
                ),
                "eef_position": tfds.features.Tensor(
                    shape=(12,), dtype=tf.float32,
                    doc=f"End-effector position command. {eef_action_desc} "
                        f"Units: radians for joint angles, meters for positions (if applicable)."
                ),
                "eef_velocity": tfds.features.Tensor(
                    shape=(12,), dtype=tf.float32,
                    doc=f"End-effector velocity command. {eef_action_desc} "
                        f"Units: rad/s for joint velocities."
                ),
                "vr_tcp_input_pose": tfds.features.Tensor(
                    shape=(14,), dtype=tf.float32,
                    doc="VR input pose from /ik_fk_result/input_pos topic. "
                        "Format: 14 floats = left[7] (pos[3] + quat[4]) + right[7] (pos[3] + quat[4]). "
                        "Position in meters, quaternion in [x, y, z, w] format. "
                        "Source: std_msgs/Float32MultiArray message. "
                        "Used for VR-based imitation learning."
                ),
            }),
            # Open-X format actions (for RT-1/RT-1-X training)
            "world_vector": tfds.features.Tensor(
                shape=(3,), dtype=tf.float32,
                doc="TCP 3D position change (relative displacement) in meters. "
                    "Computed as delta from previous step's TCP position using Forward Kinematics. "
                    "Format: [x, y, z] relative to base_link. "
                    "Compatibility mode: uses left arm value for single-arm model compatibility. "
                    "Used by Open-X models (RT-1, RT-1-X) for training and inference. "
                    "Range: typically [-2.0, 2.0] meters for RT-1-X."
            ),
            "rotation_delta": tfds.features.Tensor(
                shape=(3,), dtype=tf.float32,
                doc="TCP 3D rotation change (relative rotation) in radians. "
                    "Computed as delta from previous step's TCP orientation using Forward Kinematics. "
                    "Format: [roll, pitch, yaw] relative to base_link. "
                    "Compatibility mode: uses left arm value for single-arm model compatibility. "
                    "Used by Open-X models (RT-1, RT-1-X) for training and inference. "
                    "Range: typically [-π/2, π/2] radians."
            ),
            "world_vector_left": tfds.features.Tensor(
                shape=(3,), dtype=tf.float32,
                doc="Left arm TCP position change (relative displacement) in meters. "
                    "Computed as delta from previous step's left arm TCP position. "
                    "Format: [x, y, z] relative to base_link. "
                    "Optional field: provides detailed left arm information for user access."
            ),
            "world_vector_right": tfds.features.Tensor(
                shape=(3,), dtype=tf.float32,
                doc="Right arm TCP position change (relative displacement) in meters. "
                    "Computed as delta from previous step's right arm TCP position. "
                    "Format: [x, y, z] relative to base_link. "
                    "Optional field: provides detailed right arm information for user access."
            ),
            "rotation_delta_left": tfds.features.Tensor(
                shape=(3,), dtype=tf.float32,
                doc="Left arm TCP rotation change (relative rotation) in radians. "
                    "Computed as delta from previous step's left arm TCP orientation. "
                    "Format: [roll, pitch, yaw] relative to base_link. "
                    "Optional field: provides detailed left arm information for user access."
            ),
            "rotation_delta_right": tfds.features.Tensor(
                shape=(3,), dtype=tf.float32,
                doc="Right arm TCP rotation change (relative rotation) in radians. "
                    "Computed as delta from previous step's right arm TCP orientation. "
                    "Format: [roll, pitch, yaw] relative to base_link. "
                    "Optional field: provides detailed right arm information for user access."
            ),
            "gripper_closedness_action": tfds.features.Tensor(
                shape=(1,), dtype=tf.float32,
                doc="Gripper open/close action (placeholder for dexterous hands). "
                    "Value: 0.0 (placeholder, not meaningful for dexterous hands). "
                    "For dexterous hands, full finger state is available in observation/state/eef_position (12 dims). "
                    "Used by Open-X models (RT-1, RT-1-X) for training and inference. "
                    "Range: [-1.0, 1.0] (but always 0.0 for dexterous hands)."
            ),
            "terminate_episode": tfds.features.Tensor(
                shape=(), dtype=tf.float32,
                doc="Episode termination flag. "
                    "Value 1.0: terminate episode (end of trajectory), 0.0: continue. "
                    "Set to 1.0 on the last step of each episode (is_last=True). "
                    "Used by Open-X models (RT-1, RT-1-X) for training and inference. "
                    "Range: [0.0, 1.0]."
            ),
        })

        step_features = tfds.features.FeaturesDict({
            "observation": observation,
            "action": action,
            "reward": tfds.features.Tensor(
                shape=(), dtype=tf.float32,
                doc="Dummy reward placeholder (always 0)."
            ),
            "discount": tfds.features.Tensor(
                shape=(), dtype=tf.float32,
                doc="Dummy discount placeholder (always 1)."
            ),
            "is_first": tfds.features.Tensor(
                shape=(), dtype=tf.bool,
                doc="True on the first aligned frame of an episode."
            ),
            "is_last": tfds.features.Tensor(
                shape=(), dtype=tf.bool,
                doc="True on the final aligned frame."
            ),
            "is_terminal": tfds.features.Tensor(
                shape=(), dtype=tf.bool,
                doc="True when the episode terminates (mirrors is_last)."
            ),
        })

        episode_metadata = tfds.features.FeaturesDict({
            "episode_id": tfds.features.Text(doc="Unique identifier derived from the bag filename."),
            "bag_path": tfds.features.Text(doc="Absolute path to the ROS bag that produced this episode."),
            "eef_type": tfds.features.Text(doc="Detected end-effector type (dexhand or leju_claw)."),
            "eef_dim": tfds.features.Tensor(shape=(), dtype=tf.int32, doc="Dimension of the EEF vectors (2 or 12)."),
            "timeline": tfds.features.Text(doc="Timeline reference used for alignment (currently 'camera')."),
            "num_steps": tfds.features.Tensor(shape=(), dtype=tf.int32, doc="Number of aligned steps in the episode."),
            "camera_info_json": tfds.features.Text(doc="Serialized sensor_msgs/CameraInfo for each camera."),
            "camera_intrinsics_json": tfds.features.Text(doc="Intrinsic parameters per camera (JSON)."),
            "sidecar_json": tfds.features.Text(doc="Original sidecar JSON with task prompts and marks."),
            "camera_extrinsics_json": tfds.features.Text(doc="Optional URDF-derived extrinsics per camera."),
            "joint_names_json": tfds.features.Text(doc="Ordered joint names if provided via URDF."),
        })

        metadata = tfds.core.MetadataDict({
            "camera_info": self._meta.get("camera_info", {}),
            "marks_by_step": self._meta.get("marks_by_step", []),
            "sidecar": self._meta.get("sidecar_meta", {}),
            "camera_extrinsics": self._meta.get("camera_extrinsics", {}),
        })

        desc, doc = _build_dataset_docs(self._meta)
        info = tfds.core.DatasetInfo(
            builder=self,
            description=desc,
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset(step_features),
                "episode_metadata": episode_metadata,
            }),
            homepage="https://github.com/google-deepmind/open_x_embodiment",
            metadata=metadata,
        )
        info.doc = doc
        return info

    def _split_generators(self, dl_manager):
        source_dir = Path(self._source_dir) if self._source_dir else Path(self.data_dir)
        meta_path = source_dir / "rlds_metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
                self._update_metadata_cache()
        shard_paths_from_meta = self._meta.get("shards", [])
        self._shards = []
        if shard_paths_from_meta:
            for shard_path in shard_paths_from_meta:
                candidate = Path(shard_path)
                if candidate.exists():
                    self._shards.append(str(candidate))
                else:
                    rel = source_dir / candidate.name
                    if rel.exists():
                        self._shards.append(str(rel))
        if not self._shards:
            dataset_name = self._meta.get("dataset_name", "delivery_openx")
            split = self._meta.get("split", "train")
            pattern = f"{dataset_name}-{split}.tfrecord-*"
            self._shards = [str(p) for p in sorted(source_dir.glob(pattern))]

        split_name = self._meta.get("split", "train")
        return {
            split_name: self._generate_examples(self._shards)
        }

    def _parse_image(self, feat, prefix: str) -> Dict:
        def _get_bytes(key: str) -> bytes:
            return feat[key].bytes_list.value[0] if key in feat and feat[key].bytes_list.value else b""

        def _get_str(key: str) -> str:
            return _get_bytes(key).decode("utf-8") if key in feat and feat[key].bytes_list.value else ""

        encoded_bytes = _get_bytes(f"{prefix}/encoded")
        format_str = _get_str(f"{prefix}/format")
        stream_raw = _get_str(f"{prefix}/stream_type") or ""
        stream_type = stream_raw.lower()
        camera = _get_str(f"{prefix}/camera")
        topic = _get_str(f"{prefix}/topic")

        img_array = None
        bit_depth = 8 if stream_type != "depth" else 16

        if encoded_bytes:
            fmt_lower = format_str.lower()
            stripped = encoded_bytes
            # Detect depth PNG even if format string is "16UC1; compressedDepth png"
            is_depth_png = (stream_type == "depth") and ("png" in fmt_lower or "compresseddepth" in fmt_lower)
            if "png" in fmt_lower:
                stripped, _ = _strip_png_padding(encoded_bytes)
                bit_depth = _infer_png_bit_depth(stripped)
            try:
                if is_depth_png:
                    target_dtype = tf.uint16 if bit_depth > 8 else tf.uint8
                    decoded = tf.io.decode_png(stripped, dtype=target_dtype, channels=0)
                    img_array = decoded.numpy()
                    if img_array.ndim == 2:
                        img_array = img_array[:, :, np.newaxis]
                    elif img_array.ndim == 3 and img_array.shape[2] > 1:
                        # Some drivers pack depth into RGB triplets; keep the first channel
                        img_array = img_array[:, :, :1]
                else:
                    decoded = tf.io.decode_image(encoded_bytes, channels=3, expand_animations=False)
                    img_array = decoded.numpy()
                    bit_depth = 8
                    if img_array.ndim == 2:
                        img_array = np.stack([img_array] * 3, axis=-1)
            except Exception as exc:
                print(f"Warning: Failed to decode image from {prefix}: {exc}")

        if img_array is None:
            # Fall back to a minimal placeholder that still encodes dtype/bit depth
            channels = 1 if stream_type == "depth" else 3
            dtype = np.uint16 if (stream_type == "depth" and bit_depth > 8) else np.uint8
            img_array = np.zeros((1, 1, channels), dtype=dtype)

        return {
            "image": img_array,
            "bit_depth": int(bit_depth),
            "format": format_str,
            "camera": camera,
            "topic": topic,
            "stream_type": stream_raw,
        }

    def _parse_tensor(self, feat, key: str, length: int) -> List[float]:
        if key in feat:
            return list(feat[key].float_list.value)
        return [0.0] * length

    def _parse_step(self, example: tf.train.Example) -> Dict:
        feat = example.features.feature

        def _get_int(key):
            return int(feat[key].int64_list.value[0]) if key in feat else 0

        def _get_float(key):
            return float(feat[key].float_list.value[0]) if key in feat else 0.0

        def _get_bool(key):
            return bool(_get_int(key))

        def _get_str(key):
            return feat[key].bytes_list.value[0].decode("utf-8") if key in feat and feat[key].bytes_list.value else ""

        num_aux = int(self._meta.get("num_aux_streams", 0))
        num_aux = max(0, min(num_aux, MAX_AUX_STREAMS))

        observation = {
            "image_primary": self._parse_image(feat, "observation/image_primary"),
            "state": {
                "joint_position": self._parse_tensor(feat, "observation/state/joint_position", 28),
                "joint_velocity": self._parse_tensor(feat, "observation/state/joint_velocity", 28),
                "joint_torque": self._parse_tensor(feat, "observation/state/joint_torque", 28),
                "eef_position": self._parse_tensor(feat, "observation/state/eef_position", 12),
                "eef_velocity": self._parse_tensor(feat, "observation/state/eef_velocity", 12),
                "eef_effort": self._parse_tensor(feat, "observation/state/eef_effort", 12),
                "tcp_position_left": self._parse_tensor(feat, "observation/state/tcp_position_left", 3),
                "tcp_position_right": self._parse_tensor(feat, "observation/state/tcp_position_right", 3),
                "tcp_orientation_left": self._parse_tensor(feat, "observation/state/tcp_orientation_left", 4),
                "tcp_orientation_right": self._parse_tensor(feat, "observation/state/tcp_orientation_right", 4),
                "vr_tcp_pose": self._parse_tensor(feat, "observation/state/vr_tcp_pose", 14),
            },
            "timestamp": _get_int("timestamp"),
            "natural_language_instruction": _get_str("observation/natural_language_instruction"),
            "natural_language_instruction_1": _get_str("observation/natural_language_instruction_1"),
            "natural_language_instruction_2": _get_str("observation/natural_language_instruction_2"),
            "natural_language_instruction_3": _get_str("observation/natural_language_instruction_3"),
            "subtask_language_instruction": _get_str("observation/subtask_language_instruction"),
            "camera_extrinsics_json": _get_str("observation/camera_extrinsics_json"),
        }
        for i in range(1, num_aux + 1):
            observation[f"image_aux_{i}"] = self._parse_image(feat, f"observation/image_aux_{i}")

        action = {
            "agent": {
                "joint_position": self._parse_tensor(feat, "action/agent/joint_position", 28),
                "joint_velocity": self._parse_tensor(feat, "action/agent/joint_velocity", 28),
                "joint_torque": self._parse_tensor(feat, "action/agent/joint_torque", 28),
                "eef_position": self._parse_tensor(feat, "action/agent/eef_position", 12),
                "eef_velocity": self._parse_tensor(feat, "action/agent/eef_velocity", 12),
                "vr_tcp_input_pose": self._parse_tensor(feat, "action/agent/vr_tcp_input_pose", 14),
            },
            # Open-X format actions
            "world_vector": self._parse_tensor(feat, "action/world_vector", 3),
            "rotation_delta": self._parse_tensor(feat, "action/rotation_delta", 3),
            "world_vector_left": self._parse_tensor(feat, "action/world_vector_left", 3),
            "world_vector_right": self._parse_tensor(feat, "action/world_vector_right", 3),
            "rotation_delta_left": self._parse_tensor(feat, "action/rotation_delta_left", 3),
            "rotation_delta_right": self._parse_tensor(feat, "action/rotation_delta_right", 3),
            "gripper_closedness_action": self._parse_tensor(feat, "action/gripper_closedness_action", 1),
            "terminate_episode": _get_float("action/terminate_episode"),
        }

        return {
            "observation": observation,
            "action": action,
            "reward": _get_float("reward"),
            "discount": _get_float("discount"),
            "is_first": _get_bool("is_first"),
            "is_last": _get_bool("is_last"),
            "is_terminal": _get_bool("is_last"),
        }

    def _generate_examples(self, shard_paths: List[str]):
        episodes: Dict[str, List] = {}
        for shard_path in shard_paths:
            dataset = tf.data.TFRecordDataset([shard_path])
            for raw in dataset:
                example = tf.train.Example()
                example.ParseFromString(raw.numpy())
                feat = example.features.feature
                episode_id = feat["episode_id"].bytes_list.value[0].decode("utf-8")
                step_index = int(feat["step_index"].int64_list.value[0])
                episodes.setdefault(episode_id, []).append((step_index, example))

        for episode_id, steps_list in episodes.items():
            steps_list.sort(key=lambda x: x[0])
            steps = [self._parse_step(ex) for _, ex in steps_list]
            first_ex = steps_list[0][1]
            feat = first_ex.features.feature

            def _meta_json(key):
                return json.dumps(self._meta.get(key, {})) if key in self._meta else ""

            camera_extrinsics_json = json.dumps(self._meta.get("camera_extrinsics", {}))
            joint_names_json = json.dumps(self._meta.get("joint_names", []))
            camera_intrinsics_json = json.dumps(self._meta.get("camera_info", {}))

            def _feat_str(name: str, default: str = "") -> str:
                if name in feat and feat[name].bytes_list.value:
                    return feat[name].bytes_list.value[0].decode("utf-8")
                return default

            episode_metadata = {
                "episode_id": episode_id,
                "bag_path": feat.get("bag_path", feat.get("episode_id")).bytes_list.value[0].decode("utf-8"),
                "eef_type": self._meta.get("eef_type", "unknown"),
                "eef_dim": self._meta.get("eef_dim", 0),
                "timeline": self._meta.get("timeline", "camera"),
                "num_steps": len(steps),
                "camera_info_json": json.dumps(self._meta.get("camera_info", {})),
                "camera_intrinsics_json": camera_intrinsics_json,
                "sidecar_json": json.dumps(self._meta.get("sidecar_meta", {})),
                "camera_extrinsics_json": camera_extrinsics_json,
                "joint_names_json": joint_names_json,
            }

            if "episode_metadata/camera_intrinsics_json" in feat:
                episode_metadata["camera_intrinsics_json"] = _feat_str("episode_metadata/camera_intrinsics_json", camera_intrinsics_json)
            if "episode_metadata/camera_extrinsics_json" in feat:
                episode_metadata["camera_extrinsics_json"] = _feat_str("episode_metadata/camera_extrinsics_json", camera_extrinsics_json)

            yield episode_id, {
                "steps": steps,
                "episode_metadata": episode_metadata,
            }
