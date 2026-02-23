import os
import numpy as np

from .utils import get_panoptic_weight_alpha, get_w1_w2_long_tail_sem

from .dataset import kitti_360


class Kitti360(kitti_360.Kitti360):
    def __init__(self, 
                 panoptic, 
                 root,
                 config_path,
                 split,
                 has_image=True,
                 has_pcd=True,
                 has_label=True):
        super(Kitti360, self).__init__(root=root, config_path=config_path, split=split, has_image=has_image, has_pcd=has_pcd, has_label=has_label)

        print("[panoptic: {}]".format(panoptic))

        # addition
        self.init_pose()

        # meta
        stuff_classes = ["ignore",
                            "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person", "bicyclist",
                            "motorcyclist",
                            "road", "parking", "sidewalk", "other-ground", "building", "fence", "vegetation", "trunk",
                            "terrain", "pole", "traffic-sign"]
        thing_dataset_id_to_contiguous_id = {k: k for i, k in enumerate([1, 2, 3, 4, 5, 6, 7, 8])}
        stuff_dataset_id_to_contiguous_id = {k: k for i, k in enumerate(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])}
        self.thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
        self.stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
        self.stuff_classes = stuff_classes
        self.ignore_label = 0
        if not panoptic:
            self.evaluator_type = "sem_seg"
        else:
            self.evaluator_type = "ade20k_panoptic_seg"

        # weight
        reweight = [1, 9.47, 1.89, 1.19, 1.13, 1.48, 1.70, 1.26, 1.0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        cls_weight = 1 / (self.cls_freq + 1e-3)
        for cl, w in enumerate(cls_weight):
            if self.data_config["learning_ignore"][cl]:
                cls_weight[cl] = 0
        # cls_weight = np.array([0, 17.1782, 49.4506, 49.0822, 45.9189, 44.9322, 49.0659, 49.6848, 49.8643, 5.3644, 31.3474, 7.2694, 41.0078, 5.5935, 11.1378, 2.8731, 37.3568, 9.1691, 43.3190, 48.0684])
        cls_alpha = np.log(1 + cls_weight)
        cls_alpha = cls_alpha / cls_alpha.max()
        cls_weight_panoptic, cls_alpha_panoptic = get_panoptic_weight_alpha(cls_weight, reweight)

        self.panoptic = panoptic

        if not self.panoptic:
            self.cls_weight = cls_weight
            self.cls_alpha = cls_alpha
        else:
            self.cls_weight = cls_weight_panoptic
            self.cls_alpha = cls_alpha_panoptic
        self.long_tail = 0.1
        self.w1, self.w2, self.long_tailed_sem = get_w1_w2_long_tail_sem(self.panoptic, self.long_tail,
                                                                         cls_weight, cls_alpha,
                                                                         cls_weight_panoptic, cls_alpha_panoptic)

    def get_frame_name(self, index):
        scene, frame = self.indices_scene[index], self.indices_frame[index]
        return "scene_{}_frame_{}".format(scene, frame)

    def load_poses(sel, pose_path):
        """ Load ground truth poses (T_w_cam0) from file.
          Args:
            pose_path: (Complete) filename for the pose file
          Returns:
            A numpy array of size nx4x4 with n poses as 4x4 transformation
            matrices
        """
        # Read and parse the poses
        poses_frame = []
        poses = []
        try:
            with open(pose_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    frame_T_w_cam0 = np.fromstring(line, dtype=float, sep=' ')
                    frame, T_w_cam0 = int(frame_T_w_cam0[0]), frame_T_w_cam0[1:]
                    T_w_cam0 = T_w_cam0.reshape(4, 4)
                    # T_w_cam0 = np.vstack((T_w_cam0, [0, 0, 0, 1]))
                    poses_frame.append(frame)
                    poses.append(T_w_cam0)

        except FileNotFoundError:
            print('Ground truth poses are not avaialble.')

        return np.array(poses_frame), np.array(poses)

    def load_calib(self, calib_path):
        """ Load calibrations (T_cam_velo) from file.
        """
        # Read and parse the calibrations
        T_cam_velo = []
        try:
            with open(calib_path, 'r') as f:
                T_cam_velo = np.fromstring(f.read(), dtype=float, sep=' ')
                T_cam_velo = T_cam_velo.reshape(3, 4)
                T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

        except FileNotFoundError:
            print('Calibrations are not avaialble.')

        return np.array(T_cam_velo)

    def init_pose(self):
        self.poses_frame = {}
        self.poses = {}

        for scene in self.scenes:
            # load poses
            pose_file = os.path.join(self.root, "data_pose", scene, "cam0_to_world.txt")
            poses_frame, poses = self.load_poses(pose_file)

            self.poses_frame[scene] = np.array(poses_frame)

            inv_frame0 = np.linalg.inv(poses[0])

            # load calibrations
            calib_file = os.path.join(self.root, "calibration", "calib_cam_to_velo.txt")
            T_velo_cam = self.load_calib(calib_file)
            T_velo_cam = np.asarray(T_velo_cam).reshape((4, 4))
            T_cam_velo = np.linalg.inv(T_velo_cam)

            # convert kitti poses from camera coord to LiDAR coord
            new_poses = []
            for pose in poses:
                new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
            self.poses[scene] = np.array(new_poses)
