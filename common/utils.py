import yaml
from permissive_dict import PermissiveDict as Dict
import numpy as np


def merge_new_config(config, new_config):
    if '_base_' in new_config:
        with open(new_config['_base_'], 'r') as f:
            if hasattr(yaml, 'FullLoader'):
                yaml_config = yaml.load(f, Loader=yaml.FullLoader)
            else:
                yaml_config = yaml.load(f)
            print(yaml_config)
        merge_new_config(config, Dict(yaml_config))

    for key, val in new_config.items():
        if not isinstance(val, dict):
            config[key] = val
            continue
        if key not in config:
            config[key] = Dict()
        merge_new_config(config[key], val)

    return config


def trans_by_pose2D(pointcloud, pose):
    """
    Transform points to current frame for dynamic object segmentation
    """
    pose2D = pose.copy()
    pose2D[2, 3] = 0
    hom_points = np.zeros((pointcloud.shape[0], 4))
    hom_points[:, :2] = pointcloud[:, :2]
    hom_points[:, 3] = 1
    points_transformed = np.linalg.inv(pose2D).dot(hom_points.T).T
    """"""
    pointcloud_new = pointcloud.copy()
    pointcloud_new[:, :2] = points_transformed[:, :2]
    return pointcloud_new


def trans_by_switch(pointcloud):
    pointcloud_new = pointcloud.copy()
    pointcloud_new[:, 0] = -pointcloud[:, 1]
    pointcloud_new[:, 1] = pointcloud[:, 0]
    return pointcloud_new


def custom_vote(data):
    # Filter out zeros
    filtered_data = data[data != 0]
    
    if filtered_data.size == 0:
        # If all values are zero, return zero or any specific label
        return 0
    else:
        # Return the most frequent non-zero value
        vals, counts = np.unique(filtered_data, return_counts=True)
        return vals[np.argmax(counts)].astype(filtered_data.dtype)
