import os
import numpy as np

from .utils import get_panoptic_weight_alpha, get_w1_w2_long_tail_sem

from .dataset import pc_processor


class SemanticKitti(pc_processor.dataset.semantic_kitti.SemanticKitti):
    def __init__(self,
                 panoptic,
                 root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 config_path,  # directory of config file
                 has_image=True,
                 has_pcd=True,
                 has_label=True):
        super(SemanticKitti, self).__init__(root=root, sequences=sequences, config_path=config_path, has_image=has_image, has_pcd=has_pcd, has_label=has_label)

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
        return self.pointcloud_files[index]

    def load_poses(sel, pose_path):
        """ Load ground truth poses (T_w_cam0) from file.
          Args:
            pose_path: (Complete) filename for the pose file
          Returns:
            A numpy array of size nx4x4 with n poses as 4x4 transformation
            matrices
        """
        # Read and parse the poses
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
            else:
                poses = np.load(pose_path)['arr_0']

        except FileNotFoundError:
            print('Ground truth poses are not avaialble.')

        return np.array(poses)

    def load_calib(self, calib_path):
        """ Load calibrations (T_cam_velo) from file.
        """
        # Read and parse the calibrations
        T_cam_velo = []
        try:
            with open(calib_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Tr:' in line:
                        line = line.replace('Tr:', '')
                        T_cam_velo = np.fromstring(line, dtype=float, sep=' ')
                        T_cam_velo = T_cam_velo.reshape(3, 4)
                        T_cam_velo = np.vstack((T_cam_velo, [0, 0, 0, 1]))

        except FileNotFoundError:
            print('Calibrations are not avaialble.')

        return np.array(T_cam_velo)

    def init_pose(self):
        """
        Get poses and transform them to LiDAR coord frame for transforming point clouds
        """
        self.poses = {}

        for seq in self.sequences:
            seq = "{0:02d}".format(int(seq))

            """
            Get poses and transform them to LiDAR coord frame for transforming point clouds
            """
            # load poses
            pose_file = os.path.join(self.root, seq, "poses.txt")
            poses = np.array(self.load_poses(pose_file))
            inv_frame0 = np.linalg.inv(poses[0])

            # load calibrations
            calib_file = os.path.join(self.root, seq, "calib.txt")
            T_cam_velo = self.load_calib(calib_file)
            T_cam_velo = np.asarray(T_cam_velo).reshape((4, 4))
            T_velo_cam = np.linalg.inv(T_cam_velo)

            # convert kitti poses from camera coord to LiDAR coord
            new_poses = []
            for pose in poses:
                new_poses.append(T_velo_cam.dot(inv_frame0).dot(pose).dot(T_cam_velo))
            self.poses[seq] = np.array(new_poses)