import os
from datetime import datetime
import numpy as np
from scipy.spatial.transform import Rotation as R
from torch.utils import data


class UMOSMPlanner(data.Dataset):
    def __init__(self, root,
                 splits, mode='terrain'):
        self.root = root
        self.splits = splits

        assert mode in ['terrain', 'livox']
        self.mode = mode

        if os.path.isdir(self.root):
            print("Dataset found: {}".format(self.root))
        else:
            raise ValueError("dataset not found: {}".format(self.root))

        self.pointcloud_files = []
        self.label_files = []
        for split in self.splits:
            pointcloud_path = os.path.join(self.root, split, "terrain")
            pointcloud_files = [os.path.join(pointcloud_path, f) for f in os.listdir(pointcloud_path) if ".bin" in f]
            self.pointcloud_files.extend(pointcloud_files)
            label_path = os.path.join(self.root, split, "randomF")
            label_files = [os.path.join(label_path, f) for f in os.listdir(label_path) if ".label" in f]
            self.label_files.extend(label_files)
        self.pointcloud_files.sort()
        self.label_files.sort()
        print("Using {} pointclouds from splits {}".format(
            len(self.pointcloud_files), self.splits))

        # load config -------------------------------------
        # get color map
        sem_color_map = {
            0: [0, 0, 0],
            1: [0, 200, 255],  # building
            2: [0, 175, 0],  # vegetation
            3: [255, 0, 255],  # road
        }
        max_sem_key = 0
        for k, v in sem_color_map.items():
            if k + 1 > max_sem_key:
                max_sem_key = k + 1
        self.sem_color_lut = np.zeros((max_sem_key + 100, 3), dtype=np.float32)
        for k, v in sem_color_map.items():
            self.sem_color_lut[k] = np.array(v, np.float32) / 255.0

        sem_color_inv_map = sem_color_map
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
        learning_map = {
            0: 0,
            1: 1,
            2: 2,  # vegetation is walkable
            3: 3,
        }
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut[k] = v
        # learning map inv
        learning_map = learning_map
        max_key = 0
        for k, v in learning_map.items():
            if k > max_key:
                max_key = k
        # +100 hack making lut bigger just in case there are unknown labels
        self.class_map_lut_inv = np.zeros((max_key + 100), dtype=np.int32)
        for k, v in learning_map.items():
            self.class_map_lut_inv[k] = v

        # compute ignore class by content ratio
        cls_content = {
            0: 1,
            1: 1,
            2: 1,
            3: 1,
        }
        content = np.zeros(len(learning_map), dtype=np.float32)
        for cl, freq in cls_content.items():
            x_cl = self.class_map_lut[cl]
            content[x_cl] += freq
        self.cls_freq = content

        self.mapped_cls_name = {
            0: "ignore",
            1: "building",
            2: "vegetation",
            3: "road",
        }

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.pointcloud_files)

    def parsePathInfoByIndex(self, index):
        path = self.pointcloud_files[index]
        # linux path
        if "\\" in path:
            # windows path
            path_split = path.split("\\")
        else:
            path_split = path.split("/")
        split = path_split[-3]
        frame_id = path_split[-1].split(".")[0]
        return split, frame_id

    def labelMapping(self, label):
        label = self.class_map_lut[label]
        return label

    @staticmethod
    def readLabel(path):
        sem_label = np.fromfile(path, dtype=np.int32)+1  # label add 1
        inst_label = np.zeros((sem_label.shape[0]), np.int32)
        return sem_label, inst_label

    def loadDataByKey(self, split, frame):
        if self.mode == 'terrain':
            pointcloud_path = os.path.join(self.root, split, "terrain", f"{frame}.bin")
            pointcloud = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 3)
            pointcloud = np.concatenate([pointcloud, np.zeros((pointcloud.shape[0], 1), dtype=pointcloud.dtype)], axis=-1)  # data add zero intensity
            label_path = os.path.join(self.root, split, "randomF", f"{frame}.label")
            sem_label, inst_label = self.readLabel(label_path)
            return pointcloud, sem_label, inst_label
        elif self.mode == 'livox':
            pointcloud_path = os.path.join(self.root, split, "livox", f"{frame}.bin")
            pointcloud = np.fromfile(pointcloud_path, dtype=np.float32).reshape(-1, 4)
            pointcloud[:, 3] /= 255  # sacale reflectivity 255 to 1
            sem_label = np.zeros((pointcloud.shape[0]), np.int32)
            inst_label = np.zeros((pointcloud.shape[0]), np.int32)
            return pointcloud, sem_label, inst_label

    def loadDataByIndex(self, index):
        split, frame = self.parsePathInfoByIndex(index)
        return self.loadDataByKey(split, frame)


class UMOSMPlannerWithPose(UMOSMPlanner):
    def __init__(self, root,
                 splits, mode='terrain'):
        super(UMOSMPlannerWithPose, self).__init__(root=root, splits=splits, mode=mode)

        self.poses = {}
        for split in self.splits:
            pose_file = os.path.join(self.root, split, "poses.txt")
            poses = self.load_poses(pose_file)

            new_poses = []
            for pose in poses:
                new_poses.append(pose)
            self.poses[split] = np.array(new_poses)

        # self.odometry = {}
        # for split in self.splits:
        #     odometry_file = os.path.join(self.root, split, "odometry.txt")
        #     odometry, _ = self.load_poses2(odometry_file)

        #     new_odometry = []
        #     for pose in odometry:
        #         new_odometry.append(pose)
        #     self.odometry[split] = np.array(new_odometry)

    def load_poses(self, pose_path):
        poses = []
        try:
            if '.txt' in pose_path:
                with open(pose_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                        T_w_cam0 = T_w_cam0.reshape(3, 4)
                        T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                        poses.append(T_w_cam0)
                    return np.array(poses)
        except FileNotFoundError:
            print('Ground truth poses are not avaialble.')

    def load_poses2(self, pose_path):
        pose = []
        time = []
        try:
            if '.txt' in pose_path:
                with open(pose_path, 'r') as f:
                    for line in f.readlines():
                        txyzxyzw = np.fromstring(line, dtype=float, sep=' ')
                        t = txyzxyzw[0]
                        xyzxyzw = txyzxyzw[1:]
                        T = np.eye(4, dtype=xyzxyzw.dtype)
                        T[:3, :3] = R.from_quat(xyzxyzw[3:]).as_matrix()
                        T[:3, 3] = xyzxyzw[:3]
                        pose.append(T)
                        time.append(datetime.fromtimestamp(float(t)))
                pose = np.array(pose)
                time = np.array(time)
                return pose, time
        except FileNotFoundError:
            print('Odometry poses are not avaialble.')

    def loadDataByKey(self, split, frame):
        pointcloud, sem_label, inst_label = super(UMOSMPlannerWithPose, self).loadDataByKey(split, frame)
        pose = self.poses[split][int(frame)]
        # odometry = self.odometry[split][int(frame)]

        points = pointcloud[:, :3]  # get xyz
        hom_points = np.ones(pointcloud.shape)
        hom_points[:, :-1] = points
        if self.mode == 'terrain':
            # points_transformed = np.linalg.inv(pose).dot(hom_points.T).T  # um
            points_transformed = hom_points  # um2
            # points_transformed = np.linalg.inv(odometry).dot(hom_points.T).T  # um4
        elif self.mode == 'livox':
            T_livox_install = np.array([
                [0.9063078,  0.0000000,  0.4226183, 0],
                [0.0000000,  1.0000000,  0.0000000, 0],
                [-0.4226183,  0.0000000,  0.9063078, 0],
                [0, 0, 0, 1],
            ])  # pitch 25 degree
            points_transformed = (T_livox_install @ hom_points.T).T
        pointcloud[:, :3] = points_transformed[:, :3]

        return pointcloud, sem_label, inst_label


if __name__ == '__main__':
    data_path = '/home/yuminghuang/dataset/um-osmplanner'
    dataset = UMOSMPlannerWithPose(data_path, ["scene 1"], mode='livox')
    pointcloud, sem_label, inst_label = dataset.loadDataByIndex(5000)
    print("pointcloud, sem_label, inst_label", pointcloud, sem_label, inst_label)
    # road_points = pointcloud[sem_label == 3]
    # road_points_z = road_points[:, 2]
    # road_bincount = np.bincount(((road_points_z - road_points_z.min()) / 0.1).astype(np.int32))
    # road_binheight = np.arange(road_bincount.shape[0]) * 0.1 + road_points_z.min()
    # print("road_binheight, road_bincount", road_binheight, road_bincount)
    z = pointcloud[:, 2]
    bincount = np.bincount(((z - z.min()) / 0.1).astype(np.int32))
    binheight = np.arange(bincount.shape[0]) * 0.1 + z.min()
    print("z.min(), z.max(), binheight, bincount", z.min(), z.max(), binheight, bincount)
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pointcloud[:, :3], dtype=np.float32))
    pcd.colors = o3d.utility.Vector3dVector(np.asarray(dataset.sem_color_lut[sem_label], dtype=np.float32))
    o3d.visualization.draw_geometries([pcd])
