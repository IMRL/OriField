import os
import numpy as np


map_name_from_segmentation_class_to_segmentation_index = {
    'Undefined': 0,
    'Car': 1,
    'Truck': 2,
    'Bus': 3,

    'Other Vehicle': 4,
    'Motorcyclist': 5,
    'Bicyclist': 6,
    'Pedestrian': 7,
    'Sign': 8,
    'Traffic Light': 9,

    'Pole': 10,

    'Construction Cone': 11,
    'Bicycle': 12,
    'Motorcycle': 13,
    'Building': 14,

    'Vegetation': 15,
    'Tree Trunk': 16,

    'Curb': 17,

    'Road': 18,

    'Lane Marker': 19,

    'Other Ground': 20,

    'Walkable': 21,

    'Sidewalk': 22,
}


class Waymo():
    def __init__(self, root,
                 version='v2.0.0',
                 split='train',
                 has_image=False,
                 has_label=True):
        self.root = root
        self.has_label = has_label
        self.has_image = has_image

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        assert version in ['v2.0.0', 'v2.0.0-mini']
        assert split in ['train', 'val', 'test']
        path = ''
        if split == 'train':
            path = 'training'
        elif split == 'val':
            path = 'validation'
        elif split == 'test':
            path = 'testing'

        self.process_root = os.path.join(self.root, "process", path)

        scenes = os.listdir(os.path.join(self.process_root, "pointcloud"))
        scenes.sort()
        scenes_frame = {}
        for scene in scenes:
            if self.has_label:
                scene_path = os.path.join(os.path.join(self.process_root, "semantic"), scene)
                frame = [f[:-4] for f in os.listdir(scene_path) if ".bin" in f]
            else:
                scene_path = os.path.join(os.path.join(self.process_root, "pointcloud"), scene)
                frame = [f[:-4] for f in os.listdir(scene_path) if ".bin" in f]
            frame.sort()
            scenes_frame[scene] = frame

        print("Totally {} scenes".format(len(scenes)))

        self.indices_scene, self.indices_frame, self.scenes_index, self.total_count = iter_scenes(scenes, scenes_frame)
        self.scenes, self.scenes_frame = scenes, scenes_frame
        print("{}: {} sample: {}".format(version, split, self.total_count))

        self.sem_color_lut = np.random.uniform(low=0.0, high=1.0, size=(len(map_name_from_segmentation_class_to_segmentation_index), 3))
        self.sem_color_lut_inv = np.random.uniform(low=0.0, high=1.0, size=(len(map_name_from_segmentation_class_to_segmentation_index), 3))
        self.inst_color_map = np.random.uniform(low=0.0, high=1.0, size=(10000, 3))

        self.class_map_lut = np.zeros(len(map_name_from_segmentation_class_to_segmentation_index), dtype=np.int32)
        for k, v in map_name_from_segmentation_class_to_segmentation_index.items():
            self.class_map_lut[v] = v
        self.class_map_lut_inv = np.zeros(len(map_name_from_segmentation_class_to_segmentation_index), dtype=np.int32)
        for k, v in map_name_from_segmentation_class_to_segmentation_index.items():
            self.class_map_lut_inv[v] = v

        self.cls_freq = np.array([9.4799e-03, 8.1538e-02, 9.6358e-03, 6.1562e-03, 4.6486e-03, 8.1054e-06,
                        2.8494e-04, 7.0363e-03, 5.8292e-03, 5.3949e-04, 1.0646e-02, 3.2906e-04,
                        1.2360e-04, 1.7970e-04, 2.9695e-01, 1.8626e-01, 1.6538e-02, 1.1925e-02,
                        2.1361e-01, 6.2803e-03, 5.5219e-03, 7.3514e-02, 5.2966e-02])

        self.mapped_cls_name = {}
        for v, k in map_name_from_segmentation_class_to_segmentation_index.items():
            self.mapped_cls_name[k] = v

    def __len__(self):
        'Denotes the total number of samples'
        return self.total_count

    def parsePathInfoByIndex(self, index):
        return f"{self.scenes.index(self.indices_scene[index])}", self.indices_frame[index]

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    def loadLabelByKey(self, scene, frame):
        sem_label = np.fromfile(os.path.join(self.process_root, "semantic", scene, f"{frame}.bin"), dtype=np.int32).reshape(-1)
        inst_label = np.fromfile(os.path.join(self.process_root, "instance", scene, f"{frame}.bin"), dtype=np.int32).reshape(-1)
        return sem_label, inst_label

    def loadLabelByIndex(self, index):
        scene, frame = self.indices_scene[index], self.indices_frame[index]
        return self.loadLabelByKey(scene, frame)

    def loadDataByKey(self, scene, frame):
        pointcloud = np.fromfile(os.path.join(self.process_root, "pointcloud", scene, f"{frame}.bin"), dtype=np.float32).reshape(-1, 4)
        if self.has_label:
            sem_label, inst_label = self.loadLabelByKey(scene, frame)
        else:
            sem_label = np.zeros((pointcloud.shape[0]), np.int32)
            inst_label = np.zeros((pointcloud.shape[0]), np.int32)
        return pointcloud, sem_label, inst_label

    def loadDataByIndex(self, index):
        scene, frame = self.indices_scene[index], self.indices_frame[index]
        return self.loadDataByKey(scene, frame)

    def loadCalibrationByKey(self, scene):
        extrinsic = np.fromfile(os.path.join(self.process_root, "calibration", "extrinsic", f"{scene}.bin"), dtype=np.float64).reshape(4, 4)
        inclination = np.fromfile(os.path.join(self.process_root, "calibration", "inclination", f"{scene}.bin"), dtype=np.float64).reshape(64)
        return extrinsic, inclination

    def loadCalibrationByIndex(self, index):
        scene = self.indices_scene[index]
        return self.loadCalibrationByKey(scene)

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
    data_path = '/home/yuminghuang/dataset/waymo-mini'
    dataset = Waymo(root=data_path, version='v2.0.0', split='train', has_image=False, has_label=True)
    pointcloud, sem_label, inst_label = dataset.loadDataByIndex(index=0)
    print("pointcloud, sem_label, inst_label", pointcloud, sem_label, inst_label)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pointcloud[:, :3], dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(dataset.sem_color_lut[sem_label], dtype=np.float32))
    o3d.visualization.draw_geometries([pcd])
