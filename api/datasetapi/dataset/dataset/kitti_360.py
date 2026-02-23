# https://github.com/autonomousvision/kitti360Scripts

import os
import yaml
import numpy as np
from torch.utils import data


class Kitti360(data.Dataset):
    def __init__(self, root,
                 config_path,
                 split,
                #  seqs,
                 has_image=True,
                 has_pcd=True,
                 has_label=True):
        self.root = root
        self.split = split
        # self.seqs = seqs
        self.has_label = has_label
        self.has_image = has_image
        self.has_pcd = has_pcd

        # check file exists
        if os.path.isfile(config_path):
            self.data_config = yaml.safe_load(open(config_path, "r"))
        else:
            raise ValueError("config file not found: {}".format(config_path))

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        pointcloud_path = os.path.join(self.root, 'data_3d_raw')
        assert split in ['val'], "validation mode only"
        scenes = os.listdir(pointcloud_path)
        scenes.sort()
        scenes_frame = {}
        for scene in scenes:
            scene_path = os.path.join(pointcloud_path, scene, "velodyne_points", "data")
            frame = [f[:-4] for f in os.listdir(scene_path) if ".bin" in f]
            frame.sort()
            scenes_frame[scene] = frame

        print("Totally {} scenes".format(len(scenes)))

        self.indices_scene, self.indices_frame, self.scenes_index, self.total_count = iter_scenes(scenes, scenes_frame)
        self.scenes, self.scenes_frame = scenes, scenes_frame
        print("{} sample: {}".format(split, self.total_count))

        # load config -------------------------------------
        # get color map
        sem_color_map = self.data_config["color_map"]
        max_sem_key = 0
        for k, v in sem_color_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_map.items():
            self.sem_color_lut[k] = np.array(v, np.float32) / 255.0

        sem_color_inv_map = self.data_config["color_map_inv"]
        max_sem_key = 0
        for k, v in sem_color_inv_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut_inv = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_inv_map.items():
            self.sem_color_lut_inv[k] = np.array(v, np.float32) / 255.0

        self.inst_color_map = np.random.uniform(
            low=0.0, high=1.0, size=(10000, 3))

        # get learning class map
        # map unused classes to used classes
        learning_map = self.data_config["learning_map"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v
        # learning map inv
        learning_map = self.data_config["learning_map_inv"]
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut_inv = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut_inv[k] = v

        # compute ignore class by content ratio
        cls_content = self.data_config["content"]
        content = np.zeros(len(self.data_config["learning_map_inv"]), dtype=np.float32)
        for cl, freq in cls_content.items():
            x_cl = self.class_map_lut[cl]
            content[x_cl] += freq
        self.cls_freq = content

        self.mapped_cls_name = self.data_config["mapped_class_name"]

    def __len__(self):
        'Denotes the total number of samples'
        return self.total_count

    def parsePathInfoByIndex(self, index):
        # return f"{self.scenes.index(self.indices_scene[index])}", self.indices_frame[index]
        return self.indices_scene[index], self.indices_frame[index]

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    def loadDataByKey(self, scene, frame):
        pointcloud_path = os.path.join(self.root, 'data_3d_raw', scene, "velodyne_points", "data", f"{frame}.bin")
        pointcloud = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
        if self.has_label:
            raise NotImplementedError()
        else:
            sem_label = np.zeros((pointcloud.shape[0]), np.int32)
            inst_label = np.zeros((pointcloud.shape[0]), np.int32)
        return pointcloud, sem_label, inst_label

    def loadLabelByIndex(self, index):
        raise NotImplementedError()

    def loadDataByIndex(self, index):
        scene, frame = self.indices_scene[index], self.indices_frame[index]
        return self.loadDataByKey(scene, frame)

    def loadImage(self, index):
        raise NotImplementedError()


def iter_scenes(scenes, scenes_frame):
    indices_scene = []
    indices_frame = []
    scenes_index = {}
    total_count = 0
    for scene in scenes:
        frame = scenes_frame[scene]
        # print("scene, frame", scene, frame)
        cnt = len(frame)
        indices_scene.extend([scene]*cnt)
        indices_frame.extend(frame)
        scenes_index[scene] = np.arange(total_count, total_count+cnt)
        total_count += cnt
    return indices_scene, indices_frame, scenes_index, total_count


if __name__ == '__main__':
    data_path = '/home/yuminghuang/dataset/kitti-360/KITTI-360'
    config_path = '../semantic-kitti.yaml'
    dataset = Kitti360(data_path,
                        "val",
                        # seqs,
                        config_path,
                        has_image=False,
                        has_pcd=True,
                        has_label=False)
    pointcloud, sem_label, inst_label = dataset.loadDataByIndex(0)
    print("pointcloud, sem_label, inst_label", pointcloud, sem_label, inst_label)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pointcloud[:, :3], dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(dataset.sem_color_lut[sem_label], dtype=np.float32))
    o3d.visualization.draw_geometries([pcd])
