#!/usr/bin/env python3
import tensorflow as tf

tfrecord_path = "/data/delivery_openx/1.0.0/train/delivery_openx-train.tfrecord-00000-of-00030"

# 读取原始记录，检查有哪些 key
dataset = tf.data.TFRecordDataset(tfrecord_path)
for raw_record in dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    features = example.features.feature
    
    print("=== TFRecord 中包含 vr 的 keys ===")
    vr_keys = [k for k in features.keys() if 'vr' in k.lower()]
    for k in sorted(vr_keys):
        feat = features[k]
        if feat.float_list.value:
            val = list(feat.float_list.value)
            print(f"  {k}: float_list len={len(val)}, values={val[:7]}...")
        elif feat.int64_list.value:
            val = list(feat.int64_list.value)
            print(f"  {k}: int64_list len={len(val)}")
        elif feat.bytes_list.value:
            print(f"  {k}: bytes_list")
        else:
            print(f"  {k}: EMPTY")
    
    if not vr_keys:
        print("  没有找到包含 'vr' 的 key！")
        print("\n=== 所有 keys ===")
        for k in sorted(features.keys()):
            print(f"  {k}")
    
    # 也检查一下 state 相关的 key
    print("\n=== observation/state 相关的 keys ===")
    state_keys = [k for k in features.keys() if 'observation/state' in k]
    for k in sorted(state_keys):
        feat = features[k]
        if feat.float_list.value:
            val = list(feat.float_list.value)
            print(f"  {k}: len={len(val)}")

print("\n=== 调试完成 ===")

