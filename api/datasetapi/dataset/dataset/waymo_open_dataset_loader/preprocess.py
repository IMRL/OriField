# https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial_v2.ipynb
# https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/v2/__init__.py

import os
import sys
import numpy as np
import tqdm
import dask.dataframe as dd
# import pandas as pd
from waymo_open_dataset import v2
import tensorflow as tf


class WaymoPreprocess():
    def __init__(self, root,
                 version='v2.0.0',
                 split='train'):
        self.root = root

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        assert version in ['v2.0.0', 'v2.0.0-mini']
        assert split in ['train', 'val', 'test']
        self.path = ''
        if split == 'train':
            self.path = 'training'
        elif split == 'val':
            self.path = 'validation'
        elif split == 'test':
            self.path = 'testing'

    def read(self, main_dir, vice_dir_list):
        main_path = os.path.join(self.root, self.path, main_dir)
        main_seqs = [os.path.join(main_path, f) for f in os.listdir(main_path) if ".parquet" in f]

        vice_seqs_list = []
        for vice_dir in vice_dir_list:
            vice_path = os.path.join(self.root, self.path, vice_dir)
            vice_seqs = [os.path.join(vice_path, f) for f in os.listdir(vice_path) if ".parquet" in f]
            assert (len(main_seqs) == len(vice_seqs))
            vice_seqs_list.append(vice_seqs)

        print("Totally {} scenes".format(len(main_seqs)))

        df_main_seqs = read_parquet_by_group(main_seqs)
        df_vice_seqs_list = []
        for vice_seqs in vice_seqs_list:
            df_vice_seqs = read_parquet_by_group(vice_seqs)
            df_vice_seqs_list.append(df_vice_seqs)

        return df_main_seqs, df_vice_seqs_list

    def merge(self, df_main_seqs, df_vice_seqs_list, merge_main):
        df_group = {}
        for i, (scene, df_main_seq) in enumerate(df_main_seqs.items()):
            print(f"merging: i {i}, scene {scene}")
            df_vice_seq_list = [df_vice_seqs[scene] for df_vice_seqs in df_vice_seqs_list]
            df = merge_main(df_main_seq, df_vice_seq_list)
            df_group[scene] = df
        return df_group

    def write(self, df_group, process_path=None, process_func=None, process_finish=None):
        if process_path is not None:
            process_root = os.path.join(self.root, process_path, self.path)
        process_dict = {}
        for i, (scene, df_main_seq) in enumerate(df_group.items()):
            for j, row in tqdm.tqdm(iter(df_main_seq.iterrows()), desc=f"scene {i}/{len(df_group.keys())}"):
                assert row["key.segment_context_name"] == scene
                if process_path is not None and process_func is not None:
                    process_func(process_dict, row, process_root)
        if process_path is not None and process_finish is not None:
            process_finish(process_dict, process_root)


def read_parquet_by_group(path):
    group = {}
    for p in path:
        group[p.split('/')[-1][:-8]] = read_parquet(p)
    return group


def read_parquet(path):
    return dd.read_parquet(path)
    # return pd.read_parquet(path)


def merge_main_key(df_main, df_vice_list):
    df_pointcloud = df_vice_list[0]
    df_calibration = df_vice_list[1]
    df_label = df_vice_list[2]

    df_pointcloud = df_pointcloud[df_pointcloud['key.laser_name'] == 1]
    # df_pointcloud = df_pointcloud.rename(columns={"key.segment_context_name": "key.segment_context_name:lidar",
    #                                     "key.frame_timestamp_micros": "key.frame_timestamp_micros:lidar"})
    df_main = df_main.merge(
        df_pointcloud,
        on=[
            'key.segment_context_name',
            'key.frame_timestamp_micros',
        ],
        how='inner',
        # left_index=True,
        # right_index=True,
    )
    df_label = df_label[df_label['key.laser_name'] == 1]
    # df_label = df_label.rename(columns={"key.segment_context_name": "key.segment_context_name:lidar_segmentation_label",
    #                                     "key.frame_timestamp_micros": "key.frame_timestamp_micros:lidar_segmentation_label",
    #                                     "key.laser_name": "key.laser_name:lidar_segmentation_label"})
    df_main = df_main.merge(
        df_label,
        on=[
            'key.segment_context_name',
            'key.frame_timestamp_micros',
            'key.laser_name',
        ],
        how='inner',
        # left_index=True,
        # right_index=True,
    )
    df_calibration = df_calibration[df_calibration['key.laser_name'] == 1]
    df_main = df_main.merge(
        df_calibration,
        on=[
            'key.segment_context_name',
            'key.laser_name',
        ],
        how='inner',
    )
    return df_main


def merge_main_lidar(df_main, df_vice_list):
    df_pointcloud = df_vice_list[0]
    df_calibration = df_vice_list[1]

    df_pointcloud = df_pointcloud[df_pointcloud['key.laser_name'] == 1]
    df_main = df_main.merge(
        df_pointcloud,
        on=[
            'key.segment_context_name',
            'key.frame_timestamp_micros',
        ],
        how='inner',
    )
    df_calibration = df_calibration[df_calibration['key.laser_name'] == 1]
    df_main = df_main.merge(
        df_calibration,
        on=[
            'key.segment_context_name',
            'key.laser_name',
        ],
        how='inner',
    )
    return df_main


def merge_main_pose(df_main, df_vice_list):
    df_pose = df_vice_list[0]

    df_main = df_main.merge(
        df_pose,
        on=[
            'key.segment_context_name',
            'key.frame_timestamp_micros',
        ],
        how='inner',
    )
    return df_main


def merge_main_calibration(df_main, df_vice_list):
    df_main = df_main[df_main['key.laser_name'] == 1]
    return df_main


def convert_range_image_label(range_image1, range_image_label_tensor1, lidar_calibration):
    points_tensor1 = v2.convert_range_image_to_point_cloud(range_image1, lidar_calibration, keep_polar_features=True)

    range_image_tensor1 = range_image1.tensor
    range_image_mask1 = range_image_tensor1[..., 0] > 0
    points_label_tensor1 = tf.gather_nd(
        range_image_label_tensor1,
        tf.compat.v1.where(range_image_mask1)
    )

    points_feature = points_tensor1.numpy()
    points_label = points_label_tensor1.numpy()

    return points_feature, points_label


def convert_range_image(range_image1, lidar_calibration):
    points_tensor1 = v2.convert_range_image_to_point_cloud(range_image1, lidar_calibration, keep_polar_features=True)

    points_feature = points_tensor1.numpy()

    return points_feature


def process_func_key(process_dict, row, process_root):
    lidar = v2.LiDARComponent.from_dict(row)
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(row)

    lidar_segmentation_label = v2.LiDARSegmentationLabelComponent.from_dict(row)
    range_image1 = lidar.range_image_return1
    range_image_label_tensor1 = lidar_segmentation_label.range_image_return1.tensor
    points_feature1, points_label1 = convert_range_image_label(range_image1, range_image_label_tensor1, lidar_calibration)
    # range_image2 = lidar.range_image_return2
    # range_image_label_tensor2 = lidar_segmentation_label.range_image_return2.tensor
    # points_feature2, points_label2 = convert_range_image_label(range_image2, range_image_label_tensor2, lidar_calibration)

    points_feature, points_label = np.concatenate([points_feature1], axis=0), np.concatenate([points_label1], axis=0)

    pointcloud = points_feature[:, [3,4,5,1]]
    sem_label = points_label[:, 1]
    inst_label = points_label[:, 0]

    scene, frame = row["key.segment_context_name"], row["key.frame_timestamp_micros"]
    pointcloud_path = os.path.join(process_root, "pointcloud_key", scene)
    semantic_path = os.path.join(process_root, "semantic", scene)
    instance_path = os.path.join(process_root, "instance", scene)
    if not os.path.exists(pointcloud_path):
        os.makedirs(pointcloud_path)
    if not os.path.exists(semantic_path):
        os.makedirs(semantic_path)
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    pointcloud.reshape((-1)).astype(np.float32).tofile(os.path.join(pointcloud_path, f"{frame}.bin"))
    sem_label.reshape((-1)).astype(np.int32).tofile(os.path.join(semantic_path, f"{frame}.bin"))
    inst_label.reshape((-1)).astype(np.int32).tofile(os.path.join(instance_path, f"{frame}.bin"))

    if "statics" not in process_dict.keys():
        S_points_feature_mean = []  # range intensity elongation x y z
        S_points_feature_std = []  # range intensity elongation x y z
        S_semantic_label_cnt = np.zeros(23, dtype=np.int32)
        total_count = np.array([0], dtype=np.int32)
        process_dict["statics"] = (S_points_feature_mean, S_points_feature_std, S_semantic_label_cnt, total_count)
    S_points_feature_mean, S_points_feature_std, S_semantic_label_cnt, total_count = process_dict["statics"]
    S_points_feature_mean.append(points_feature.mean(0))
    S_points_feature_std.append(points_feature.std(0))
    label, count = np.unique(sem_label, return_counts=True)
    for lbl, cnt in zip(label, count):
        S_semantic_label_cnt[lbl] += cnt
    total_count[:] += 1


def process_finish_key(process_dict, process_root):
    S_points_feature_mean, S_points_feature_std, S_semantic_label_cnt, total_count = process_dict["statics"]
    statics = f"""
statics: 

mapped: 
points_feature_mean: {S_points_feature_mean}
points_feature_std: {S_points_feature_std}
semantic_label_cnt: {S_semantic_label_cnt}
total_count: {total_count}
reduced: 
points_feature_mean: {np.stack(S_points_feature_mean, axis=0).mean(0)}
points_feature_std: {np.stack(S_points_feature_std, axis=0).mean(0) + np.stack(S_points_feature_mean, axis=0).std(0)}
semantic_label_cnt: {S_semantic_label_cnt / S_semantic_label_cnt.sum()}
total_count: {total_count}
    """
    with open(os.path.join(process_root, "preprocess_waymo_statics.txt"), "w") as f:
        f.write(statics)
    print(statics)


def process_func_lidar(process_dict, row, process_root):
    lidar = v2.LiDARComponent.from_dict(row)
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(row)

    range_image1 = lidar.range_image_return1
    points_feature1 = convert_range_image(range_image1, lidar_calibration)
    # range_image2 = lidar.range_image_return2
    # points_feature2 = convert_range_image(range_image2, lidar_calibration)

    points_feature = np.concatenate([points_feature1], axis=0)

    pointcloud = points_feature[:, [3,4,5,1]]

    scene, frame = row["key.segment_context_name"], row["key.frame_timestamp_micros"]
    pointcloud_path = os.path.join(process_root, "pointcloud", scene)
    if not os.path.exists(pointcloud_path):
        os.makedirs(pointcloud_path)
    pointcloud.reshape((-1)).astype(np.float32).tofile(os.path.join(pointcloud_path, f"{frame}.bin"))


def process_func_pose(process_dict, row, process_root):
    vehicle_pose = v2.VehiclePoseComponent.from_dict(row)
    pose = vehicle_pose.world_from_vehicle.transform.reshape(4, 4)
    
    scene, frame = row["key.segment_context_name"], row["key.frame_timestamp_micros"]
    pose_path = os.path.join(process_root, "pose", scene)
    if not os.path.exists(pose_path):
        os.makedirs(pose_path)
    pose.reshape((-1)).astype(np.float32).tofile(os.path.join(pose_path, f"{frame}.bin"))


def process_func_calibration(process_dict, row, process_root):
    lidar_calibration = v2.LiDARCalibrationComponent.from_dict(row)
    extrinsic = lidar_calibration.extrinsic.transform.reshape(4, 4)
    inclination = lidar_calibration.beam_inclination.values
    
    scene = row["key.segment_context_name"]
    extrinsic_path = os.path.join(process_root, "calibration", "extrinsic")
    inclination_path = os.path.join(process_root, "calibration", "inclination")
    if not os.path.exists(extrinsic_path):
        os.makedirs(extrinsic_path)
    extrinsic.reshape((-1)).astype(np.float64).tofile(os.path.join(extrinsic_path, f"{scene}.bin"))
    if not os.path.exists(inclination_path):
        os.makedirs(inclination_path)
    inclination.reshape((-1)).astype(np.float64).tofile(os.path.join(inclination_path, f"{scene}.bin"))


if __name__ == '__main__':
    assert len(sys.argv) == 2, "usage: python preprocess.py /path/to/your/waymo/dataset"
    data_path = sys.argv[1]

    dataset = WaymoPreprocess(root=data_path, version='v2.0.0', split='train')

    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["lidar", "lidar_calibration", "lidar_segmentation"])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_key)
    dataset.write(df_group, process_path="process", process_func=process_func_key, process_finish=process_finish_key)
    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["lidar", "lidar_calibration"])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_lidar)
    dataset.write(df_group, process_path="process", process_func=process_func_lidar)
    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["vehicle_pose"])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_pose)
    dataset.write(df_group, process_path="process", process_func=process_func_pose)
    df_main_seqs, df_vice_seqs_list = dataset.read("lidar_calibration", [])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_calibration)
    dataset.write(df_group, process_path="process", process_func=process_func_calibration)

    dataset = WaymoPreprocess(root=data_path, version='v2.0.0', split='val')

    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["lidar", "lidar_calibration", "lidar_segmentation"])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_key)
    dataset.write(df_group, process_path="process", process_func=process_func_key, process_finish=process_finish_key)
    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["lidar", "lidar_calibration"])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_lidar)
    dataset.write(df_group, process_path="process", process_func=process_func_lidar)
    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["vehicle_pose"])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_pose)
    dataset.write(df_group, process_path="process", process_func=process_func_pose)
    df_main_seqs, df_vice_seqs_list = dataset.read("lidar_calibration", [])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_calibration)
    dataset.write(df_group, process_path="process", process_func=process_func_calibration)

    dataset = WaymoPreprocess(root=data_path, version='v2.0.0', split='test')

    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["lidar", "lidar_calibration", "lidar_segmentation"])
    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["lidar", "lidar_calibration"])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_lidar)
    dataset.write(df_group, process_path="process", process_func=process_func_lidar)
    df_main_seqs, df_vice_seqs_list = dataset.read("stats", ["vehicle_pose"])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_pose)
    dataset.write(df_group, process_path="process", process_func=process_func_pose)
    df_main_seqs, df_vice_seqs_list = dataset.read("lidar_calibration", [])
    df_group = dataset.merge(df_main_seqs, df_vice_seqs_list, merge_main_calibration)
    dataset.write(df_group, process_path="process", process_func=process_func_calibration)
