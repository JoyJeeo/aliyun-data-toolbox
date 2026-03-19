# rosbag → TFDS 转换工具包

## 目录/脚本作用
- `Dockerfile`：基于 `osrf/ros:noetic-desktop` 构建运行环境，构建阶段即安装依赖。
- `setup_container.sh`：在容器内安装 TensorFlow/TFDS、rosbag、Drake（用于 FK，失败则 TCP 置零）等依赖。
- `tfds_adapter_delivery_openx.py`：加载/校验生成的 TFDS 数据集（本地 builder）。
- `tools/run_conversion.sh`：批量调用 `rosbag_to_tfds.py` 的入口脚本。
- `tools/rosbag_to_tfds.py`：核心转换脚本（一个 bag → 一个 TFDS 分片）。
- 支撑模块（被 rosbag_to_tfds.py 引用）：`tfds_builder_delivery_openx.py`, `rosbag_reader.py`, `synchronization.py`, `sidecar_utils.py`, `tf_reader.py`, `urdf_utils.py`, `drake_fk_utils.py`, `writer.py`, `rosbag_to_openx.py`（提供姿态增量、指令构建等工具函数）。

## 构建镜像
```bash
docker build -t rosbag2tfds:latest .
```
构建过程中会执行 `setup_container.sh` 安装依赖；如果网络受限，需要保证构建阶段可翻墙。

## 运行方式（示例）
假设本机有 `rosbag/` 目录和输出目录 `tfds_out/`：
```bash
docker run --rm -it \
  -v /path/to/rosbag:/work/rosbag:ro \
  -v /path/to/tfds_out:/work/tfds_out \
  rosbag2tfds:latest \
  /bin/bash -lc "cd /work && tools/run_conversion.sh --input_root /work/rosbag --output_dir /work/tfds_out"
```

## 可传参（常用）
`tools/run_conversion.sh` 透传到 `rosbag_to_tfds.py` 的参数，可用环境变量或直接传：
- `--input_root`（必填）：包含若干 bag 子目录的路径（每个子目录内有 .bag 及 sidecar json）。
- `--output_dir`（必填）：TFDS 输出根目录。
- 摄像头相关：`--main_rgb_topic`、`--rgb_topics`、`--depth_topics`、`--camera_info_topics`。
- 对齐/时间线：`--timeline`（默认为 camera），`--clip_to_marks`。
- TF 相关：`--tf_topic`（默认 /tf）、`--base_frame`、`--tcp_frame_left/right`。
- URDF：`--urdf`（用于 FK 和相机外参；缺失则 TCP 置零、外参为空）。
- EEF：`--eef`（auto/leju_claw/dexhand）。
注意：长 episode 会直接写入单条 TFDS Example；如极端长可考虑 clip 或分段。
注意：将脚本部署到平台上时，传参信息与之前rosbag2rlds基本一致，但是务必检查原始数据中的各个topic的名字是否匹配。 

## 依赖关系说明
- 运行时依赖：ROS1 Noetic（rosbag 模块），TensorFlow/TFDS，numpy/scipy，protobuf<4.22。
- 可选依赖：Drake（pydrake）用于 FK；缺失时 TCP 位姿置零并有提示。
- 脚本间调用：`run_conversion.sh` → `rosbag_to_tfds.py` → 支撑模块（reader/对齐/URDF/TF 解析等） → `tfds_builder_delivery_openx.py` 定义的 features。
- 校验/加载：生成后可用 `tfds_adapter_delivery_openx.py` 或自定义脚本（如检查 dtype/format 的脚本）验证 RGB=jpeg/uint8，Depth=png/uint16。
