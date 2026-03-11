import os
import json
import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

def find_lerobot_dirs(root_dir):
    lerobot_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "lerobot" in dirnames:
            lerobot_dirs.append(os.path.join(dirpath, "lerobot"))
    return lerobot_dirs


def get_total_episodes(dataset_dir):
    info_path = Path(dataset_dir) / "meta" / "info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        total_episodes = info.get("total_episodes")
        if isinstance(total_episodes, int) and total_episodes > 0:
            return total_episodes

    return len(list((Path(dataset_dir) / "data").glob("**/episode_*.parquet")))


def iter_metadata_entries(metadata):
    if isinstance(metadata, list):
        for idx, item in enumerate(metadata):
            if isinstance(item, dict):
                yield idx, item
        return

    if isinstance(metadata, dict):
        if "episode_id" in metadata:
            yield 0, metadata
            return

        keys = list(metadata.keys())
        if keys and all(str(key).isdigit() and isinstance(metadata[key], dict) for key in keys):
            for key in sorted(keys, key=lambda x: int(x)):
                yield int(key), metadata[key]
            return

        # Treat a normal JSON object as one metadata record.
        yield 0, metadata


def merge_metadata_json(lerobot_dirs, output_dir):
    merged_metadata = {}
    episode_offset = 0
    found_any_metadata = False

    for dataset_dir in sorted(lerobot_dirs):
        metadata_path = Path(dataset_dir) / "metadata.json"
        total_episodes = get_total_episodes(dataset_dir)

        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                for local_idx, item in iter_metadata_entries(metadata):
                    new_idx = episode_offset + local_idx
                    merged_item = dict(item)
                    merged_item["episode_id"] = new_idx
                    merged_metadata[str(new_idx)] = merged_item
                found_any_metadata = True
            except Exception as e:
                print(f"[WARNING] 读取 metadata.json 失败 ({metadata_path}): {e}")

        episode_offset += total_episodes

    if not found_any_metadata:
        return

    output_path = Path(output_dir) / "metadata.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merged_metadata, f, indent=2, ensure_ascii=False)
    print(f"✅ 已合并 metadata.json: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        default="/home/leju_wxc/wxc_ws/leju_kuavo/temp",
        type=str,
        required=False,
        help="Path to the root directory containing lerobot datasets",
    )
    parser.add_argument(
        "--output_dir",
        default="/home/leju_wxc/wxc_ws/leju_kuavo/temp/kuavo-merged-dataset",
        type=str,
        required=False,
        help="Output directory for merged dataset",
    )
    parser.add_argument(
        "--output_repo_id",
        default="lerobot/kuavo-merged-dataset",
        type=str,
        required=False,
        help="Output repo id for merged dataset",
    )
    args = parser.parse_args()

    lerobot_dirs = find_lerobot_dirs(args.input_dir)
    datasets = [LeRobotDataset(d) for d in lerobot_dirs]
    merged = merge_datasets(
        datasets,
        output_repo_id=args.output_repo_id,
        output_dir=args.output_dir
    )
    merge_metadata_json(lerobot_dirs, args.output_dir)
