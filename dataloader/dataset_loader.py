import os
import time
import logging
import yaml
from permissive_dict import PermissiveDict as Dict
from matplotlib import cm

import numpy as np
from scipy.spatial.distance import cdist
from tslearn.metrics import dtw_path
import rdp
import open3d as o3d
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

import matplotlib as mpl
import matplotlib.pyplot as plt

from lidardet.ops.lidar_bev import bev

from common.timer import Timer
from common.traversibility import padding_and_traversable, estimate_frontier_and_skeleton, nearest_cluster
from common.edt_tangent import edt_with_tangent
from common.polar import *
from common.visulization import Visulizater
from common.peroidic import peroidic_np
from common import planning
from common.ngm import smooth_tangent, brocast_tangent, smooth_and_brocast_tangent, pierce_tangent, normalize_distance
from common.coordinate import Convertor
from common.curve import curve_by_curviness, curve_by_control
from common.utils import merge_new_config, trans_by_pose2D, trans_by_switch, custom_vote
from common.envs import *


def point_to_segment_distance(point, segment_start, segment_end):
    # Vector from segment start to end
    segment_vector = segment_end - segment_start
    # Vector from segment start to the point
    point_vector = point - segment_start
    # Calculate the projection of the point onto the line defined by the segment
    segment_length_squared = np.dot(segment_vector, segment_vector)
    if segment_length_squared == 0:
        # The segment start and end points are the same
        return np.linalg.norm(point_vector)

    # Projection factor
    t = np.dot(point_vector, segment_vector) / segment_length_squared
    # Clamp t to the range [0, 1] to find the closest point on the segment
    t = max(0, min(1, t))
    # The closest point on the segment
    closest_point = segment_start + t * segment_vector
    # Distance from the point to the closest point on the segment
    distance = np.linalg.norm(point - closest_point)
    return distance


class NGMInterfance():
    @property
    def dataset_len(self):
        return len(self.dataset)
    @property
    def dataset_sem_color_lut(self):
        return self.dataset.sem_color_lut
    @property
    def dataset_poses(self):
        return self.dataset.poses
    @property
    def dataset_root(self):
        return self.dataset.root
    def dataset_loadDataByKey(self, seq_id, frame_id):
        return self.dataset.loadDataByKey(seq_id, frame_id)
    def dataset_parsePathInfoByIndex(self, idx):
        return self.dataset.parsePathInfoByIndex(idx)


class NGMDataset(Dataset):

    def __init__(self, cfg, logger=None):
        super(NGMDataset, self).__init__()

        if logger:
            logger.info(
                'Loading TrajectoryPrediction dataset with {} samples'.format(self.dataset_len))
        self.sensor_height = cfg.sensor_height
        self.res = cfg.voxel_size[0]  # 0.16 m / pixel
        self.res_z = cfg.voxel_size[2]  # 2.0 m / pixel
        xmin, ymin, zmin, xmax, ymax, zmax = cfg.point_cloud_range
        self.lidar_range_np = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
        self.lidar_range = {'Left': xmin, 'Right': xmax, 'Front': ymax, 'Back': ymin, 'Bottom': zmin, 'Top': zmax}
        self.map_h = self.lidar_range['Front'] - self.lidar_range['Back']
        self.map_w = self.lidar_range['Right'] - self.lidar_range['Left']
        self.map_c = self.lidar_range['Top'] - self.lidar_range['Bottom']
        self.img_h = round(self.map_h / self.res)
        self.img_w = round(self.map_w / self.res)
        self.img_c = round(self.map_c / self.res_z)
        print("self.img_h, self.img_w, self.img_c", self.img_h, self.img_w, self.img_c)

        self.convertor = Convertor(img_h=self.img_h, img_w=self.img_w, res=self.res, lidar_range=self.lidar_range)

        self.trajectory_colors = cm.get_cmap('viridis', 10)
        self.image_transform = ToTensor()

        self.colormap = np.zeros_like(self.dataset_sem_color_lut)
        self.colormap[..., 0] = self.dataset_sem_color_lut[..., 2]
        self.colormap[..., 1] = self.dataset_sem_color_lut[..., 1]
        self.colormap[..., 2] = self.dataset_sem_color_lut[..., 0]
        self.visulizater = Visulizater(self.colormap)

        dataset_seq_id_list = list(self.dataset_poses.keys())
        dataset_seq_id_list.sort()
        print("dataset_seq_id_list", dataset_seq_id_list)
        self.full_trajectorys = {}
        for seq_id in dataset_seq_id_list:
            self.full_trajectorys[seq_id] = self.dataset_poses[seq_id][:, :2, 3]
        if self.is_deploy:
            self.full_guidances = {}
            self.full_correspondances = {}
            self.full_xmeters = {}
            for seq_id in dataset_seq_id_list:
                guidance_poses = self.load_guidances(os.path.join(self.dataset_root, seq_id, "guidance_osm.txt"))
                if guidance_poses is not None:
                    self.full_guidances[seq_id] = guidance_poses[:, :2]
                    self.full_xmeters[seq_id] = np.concatenate([np.array([0]), np.cumsum(np.linalg.norm(np.diff(self.full_guidances[seq_id], axis=0), axis=1))]) // 5
                    corr_list, dist = dtw_path(self.full_trajectorys[seq_id], self.full_guidances[seq_id])
                    corr_map = {}
                    for corr in corr_list:
                        ego_id = corr[0]
                        correspondance_id = corr[1]
                        if ego_id not in corr_map.keys():
                            corr_map[ego_id] = []
                        corr_map[ego_id].append(correspondance_id)
                    # print("corr_list, dist, corr_map", corr_list, dist, corr_map)
                    self.full_correspondances[seq_id] = np.full((len(corr_map.keys())), -1)
                    for ego_id, correspondance_ids in corr_map.items():
                        correspondance_id = correspondance_ids[np.linalg.norm(self.full_guidances[seq_id][correspondance_ids] - self.full_trajectorys[seq_id][ego_id], axis=-1).argmin()]
                        self.full_correspondances[seq_id][ego_id] = correspondance_id
                    assert ((self.full_correspondances[seq_id] < 0) | (self.full_correspondances[seq_id] >= self.full_guidances[seq_id].shape[0])).sum() == 0, \
                        f"ill correspondances for scene {seq_id} {self.full_trajectorys[seq_id]} {self.full_guidances[seq_id]} {self.full_correspondances[seq_id]}"

        if PREPROCESS_SCANS_TRAINING_ACC == 'mapping':
            self.built_map = {}
            for seq_id in dataset_seq_id_list:
                xyzi = o3d.t.io.read_point_cloud(os.path.join(self.dataset_root, seq_id, "terrainMapPGO.ply"))
                self.built_map[seq_id] = np.concatenate([xyzi.point["positions"][:, :].numpy(), xyzi.point["scalar_intensity"][:, :].numpy()], axis=-1)

    def load_guidances(self, guidance_path):
        try:
            with open(guidance_path, 'r') as f:
                guidances = []
                lines = f.readlines()
                for line in lines:
                    guidances.append(np.fromstring(line, dtype=float, sep=' '))
                return np.stack(guidances, axis=0)
        except FileNotFoundError:
            return None

    def __len__(self):
        return self.dataset_len

    def rgb_label_bev(self, points, labels):
        bev_map = bev.rgb_label_map(points[:, :4], labels[:, None], self.lidar_range_np, self.res)
        bev_map = bev_map.reshape((self.img_h, self.img_w, -1)).copy()
        return bev_map

    def fall_in_mask(self, points):
        return (points[:, 0] >= self.lidar_range['Left']) & (points[:, 0] < self.lidar_range['Right']) & (points[:, 1] > self.lidar_range['Back']) & (points[:, 1] <= self.lidar_range['Front'])

    def pad_bev_center(self, label_bev, pad=20, flood=(0.5,0.5)):
        img_h, img_w = label_bev.shape
        x, y = np.mgrid[:img_h, :img_w]
        o = np.array([int(img_h*flood[0]), int(img_w*flood[1])])
        dis = ((x - o[0]) ** 2 + (y - o[1]) ** 2) ** .5
        mask = (dis <= pad) & (label_bev == 0)
        label_bev[mask] = self.dataset_label_traversable

    def append_scan(self, acc_points, acc_sem_label, acc_inst_label, pose_0, pose_i, seq_id_i, frame_id_i):
        pointcloud_i, sem_label_i, inst_label_i = self.dataset_loadDataByKey(seq_id_i, frame_id_i)

        points_i = pointcloud_i[:, :3]  # get xyz
        """
        Transform points to current frame for dynamic object segmentation
        """
        hom_points_i = np.ones(pointcloud_i.shape)
        hom_points_i[:, :-1] = points_i
        points_transformed_i = np.linalg.inv(pose_0).dot(pose_i).dot(hom_points_i.T).T
        """"""
        pointcloud_i[:, :3] = points_transformed_i[:, :3]

        acc_points.append(pointcloud_i)
        acc_sem_label.append(sem_label_i)
        acc_inst_label.append(inst_label_i)
        return acc_points, acc_sem_label, acc_inst_label

    def clear_points(self, points, sem_label, inst_label):
        points = trans_by_switch(points)
        points[:, 2] += self.sensor_height
        bev_mask = self.fall_in_mask(points)
        points, sem_label, inst_label = points[bev_mask], sem_label[bev_mask], inst_label[bev_mask]
        return points, sem_label, inst_label

    def estimate_bev(self, acc_points, acc_sem_label, acc_inst_label, label_fusion=True):
        if label_fusion:
            single_rgb_bev_list = []
            single_label_bev_list = []
            for points, sem_label, inst_label in zip(acc_points, acc_sem_label, acc_inst_label):
                points, sem_label, inst_label = self.clear_points(points, sem_label, inst_label)
                single_rgb_label_bev = self.rgb_label_bev(points, sem_label)
                single_rgb_bev = single_rgb_label_bev[..., :-1]
                single_label_bev = single_rgb_label_bev[..., -1].astype(np.int32)
                single_rgb_bev_list.append(single_rgb_bev)
                single_label_bev_list.append(single_label_bev)
            single_rgb_bev_list = np.stack(single_rgb_bev_list, axis=0)
            single_label_bev_list = np.stack(single_label_bev_list, axis=0)
            fusion_rgb_bev = np.max(single_rgb_bev_list, axis=0)
            fusion_label_bev = np.apply_along_axis(custom_vote, 0, single_label_bev_list)

        acc_points, acc_sem_label, acc_inst_label = np.concatenate(acc_points, axis=0), np.concatenate(acc_sem_label, axis=0), np.concatenate(acc_inst_label, axis=0)
        acc_points, acc_sem_label, acc_inst_label = self.clear_points(acc_points, acc_sem_label, acc_inst_label)
        rgb_label_bev = self.rgb_label_bev(acc_points, acc_sem_label)
        rgb_bev = rgb_label_bev[..., :-1]
        label_bev = rgb_label_bev[..., -1].astype(np.int32)

        final_rgb_bev = rgb_bev
        if label_fusion:
            final_label_bev = fusion_label_bev
        else:
            final_label_bev = label_bev
        return final_rgb_bev, final_label_bev

    def estimate_traversable(self, label_bev):
        label_erosion, label_padded, binary_traversable = padding_and_traversable(label_bev)
        label_traversable = np.where(binary_traversable, self.dataset_label_traversable, 0)
        return label_erosion, label_padded, binary_traversable, label_traversable

    def scans_single(self, idx):
        scans_n = 0

        seq_id_0, frame_id_0 = self.dataset_parsePathInfoByIndex(idx)
        pose_0 = self.dataset_poses[seq_id_0][int(frame_id_0)]

        trajectory = self.full_trajectorys[seq_id_0][:int(frame_id_0)+1]
        trajectory = trans_by_pose2D(trajectory, pose_0)
        trajectory = trans_by_switch(trajectory)
        trajectory_traj, trajectory_points, trajectory_idxs, trajectory_hit = self.cut_traj(trajectory, int(frame_id_0))
        acc_points, acc_sem_label, acc_inst_label = [], [], []
        for i in (np.arange(1, scans_n+1) * (len(trajectory_idxs) - 1) / (scans_n + 1)).astype(np.int64).tolist() + [len(trajectory_idxs) - 1]:
            trajectory_id = trajectory_idxs[i]
            seq_id_i = seq_id_0
            frame_id_i = str(trajectory_id).zfill(len(frame_id_0))
            pose_i = self.dataset_poses[seq_id_i][int(frame_id_i)]
            acc_points, acc_sem_label, acc_inst_label = self.append_scan(acc_points, acc_sem_label, acc_inst_label, pose_0, pose_i, seq_id_i, frame_id_i)
        return acc_points, acc_sem_label, acc_inst_label

    def scans_deploy(self, idx):
        scans_n = self.dataset_scans_deploy_n

        seq_id_0, frame_id_0 = self.dataset_parsePathInfoByIndex(idx)
        pose_0 = self.dataset_poses[seq_id_0][int(frame_id_0)]

        trajectory = self.full_trajectorys[seq_id_0][:int(frame_id_0)+1]
        trajectory = trans_by_pose2D(trajectory, pose_0)
        trajectory = trans_by_switch(trajectory)
        trajectory_traj, trajectory_points, trajectory_idxs, trajectory_hit = self.cut_traj(trajectory, int(frame_id_0))
        acc_points, acc_sem_label, acc_inst_label = [], [], []
        for i in (np.arange(1, scans_n+1) * (len(trajectory_idxs) - 1) / (scans_n + 1)).astype(np.int64).tolist() + [len(trajectory_idxs) - 1]:
            trajectory_id = trajectory_idxs[i]
            seq_id_i = seq_id_0
            frame_id_i = str(trajectory_id).zfill(len(frame_id_0))
            pose_i = self.dataset_poses[seq_id_i][int(frame_id_i)]
            acc_points, acc_sem_label, acc_inst_label = self.append_scan(acc_points, acc_sem_label, acc_inst_label, pose_0, pose_i, seq_id_i, frame_id_i)
        return acc_points, acc_sem_label, acc_inst_label

    def scans_training_temporal(self, idx):
        scans_n = self.dataset_scans_training_n

        seq_id_0, frame_id_0 = self.dataset_parsePathInfoByIndex(idx)
        pose_0 = self.dataset_poses[seq_id_0][int(frame_id_0)]

        trajectory = self.full_trajectorys[seq_id_0].copy()
        trajectory = trans_by_pose2D(trajectory, pose_0)
        trajectory = trans_by_switch(trajectory)
        trajectory_traj, trajectory_points, trajectory_idxs, trajectory_hit = self.cut_traj(trajectory, int(frame_id_0))
        acc_points, acc_sem_label, acc_inst_label = [], [], []
        for i in (np.arange(1, scans_n+1) * (len(trajectory_idxs) - 1) / (scans_n + 1)).astype(np.int64).tolist() + [len(trajectory_idxs) - 1]:
            trajectory_id = trajectory_idxs[i]
            seq_id_i = seq_id_0
            frame_id_i = str(trajectory_id).zfill(len(frame_id_0))
            pose_i = self.dataset_poses[seq_id_i][int(frame_id_i)]
            acc_points, acc_sem_label, acc_inst_label = self.append_scan(acc_points, acc_sem_label, acc_inst_label, pose_0, pose_i, seq_id_i, frame_id_i)
        return acc_points, acc_sem_label, acc_inst_label

    def scans_training_mapping(self, idx):
        seq_id_0, frame_id_0 = self.dataset_parsePathInfoByIndex(idx)
        pose_0 = self.dataset_poses[seq_id_0][int(frame_id_0)]

        croping_radius = self.res * (self.img_h ** 2 + self.img_w ** 2) ** .5 / 2
        croping_origin = pose_0[:2, 3]
        croping_shift = self.built_map[seq_id_0][:, :2] - croping_origin[None]  # N 2 - 1 2
        crop = self.built_map[seq_id_0][(-croping_radius <= croping_shift[:, 0]) & (croping_shift[:, 0] <= croping_radius) & (-croping_radius <= croping_shift[:, 1]) & (croping_shift[:, 1] <= croping_radius)]

        hom_points = np.ones(crop.shape)
        hom_points[:, :-1] = crop[:, :3]
        transformed_points = np.linalg.inv(pose_0).dot(hom_points.T).T
        crop[:, :3] = transformed_points[:, :3]

        points, sem_label, inst_label = crop[:, :3], np.rint(crop[:, 3])+1, np.zeros((crop.shape[0], 1), dtype=np.int32)
        acc_points, acc_sem_label, acc_inst_label = [points], [sem_label], [inst_label]
        return acc_points, acc_sem_label, acc_inst_label

    def cut_traj(self, full_traj, color):
        backward_color = color
        while True:
            if backward_color == 0 or not self.fall_in_mask(full_traj[backward_color-1][None])[0]:
                break
            backward_color -= 1
        forward_color = color
        while True:
            if forward_color == len(full_traj)-1 or not self.fall_in_mask(full_traj[forward_color+1][None])[0]:
                break
            forward_color += 1
        traj_idxs = np.arange(backward_color,forward_color+1)
        traj_traj = full_traj[traj_idxs]
        traj_points = self.convertor.car2matrix(traj_traj)
        traj_hit = color - backward_color
        return traj_traj, traj_points, traj_idxs, traj_hit

    def radius_traj(self, full_traj, color):
        meters = min(self.img_h, self.img_w) * self.res / 2
        backward_color = color
        while True:
            if backward_color == 0 or np.linalg.norm(full_traj[backward_color-1] - full_traj[color]) > meters:
                break
            backward_color -= 1
        forward_color = color
        while True:
            if forward_color == len(full_traj)-1 or np.linalg.norm(full_traj[forward_color+1] - full_traj[color]) > meters:
                break
            forward_color += 1
        traj_idxs = np.arange(backward_color,forward_color+1)
        traj_traj = full_traj[traj_idxs]
        traj_points = self.convertor.car2matrix(traj_traj)
        traj_hit = color - backward_color
        return traj_traj, traj_points, traj_idxs, traj_hit

    def span_traj(self, full_traj, color, length):
        colors = np.arange(np.max([color-length,0]),np.min([color+length+1,len(full_traj)]))
        colors_traj = full_traj[colors]
        traj_idxs = colors[self.fall_in_mask(colors_traj)]
        traj_traj = full_traj[traj_idxs]
        traj_points = self.convertor.car2matrix(traj_traj)
        return traj_traj, traj_points, traj_idxs

    def rdp_instruction(self, traj, color):
        rdp_mask = rdp.rdp(traj, epsilon=1, return_mask=True, dist=point_to_segment_distance)
        rdp_traj = traj[rdp_mask]
        rdp_diff = np.diff(rdp_traj, axis=0)
        rdp_motion  = np.arctan2(rdp_diff[1:, 1], rdp_diff[1:, 0]) - np.arctan2(rdp_diff[:-1, 1], rdp_diff[:-1, 0])
        if rdp_traj.shape[0] < 2:
            rdp_motion = np.full(rdp_traj.shape[0], 0.)
        else:
            rdp_motion = np.concatenate([np.array([0.]), rdp_motion, np.array([0.])])
        traj_motion = np.full(traj.shape[0], 0.)
        traj_motion[rdp_mask] = rdp_motion
        rdp_pilot = np.arctan2(rdp_diff[:, 1], rdp_diff[:, 0])
        if rdp_traj.shape[0] < 2:
            rdp_pilot = np.full(rdp_traj.shape[0], 0.)
        else:
            rdp_pilot = np.concatenate([rdp_pilot[0][None], rdp_pilot])
        traj_pilot = np.full(traj.shape[0], 0.)
        traj_pilot[rdp_mask] = rdp_pilot

        forward_homo = (~rdp_mask)[color:]
        backward_homo = (~rdp_mask)[:color+1][::-1]
        if (~forward_homo).sum() > 0:
            forward_cl = color + np.transpose(np.nonzero(~forward_homo))[0, 0]
            forward_cu = color + np.transpose(np.nonzero(~forward_homo))[-1, 0]
        else:
            forward_cl = len(traj) - 1
            forward_cu = len(traj) - 1
        crux_ = traj_pilot[forward_cu] - traj_pilot[forward_cl]
        crux_ = peroidic_np(crux_, norm=True, abs=False)  # (-1, 1]
        crux_ = crux_ * np.pi
        if crux_ >= 0 and crux_ < np.pi/4:
            return 0
        elif crux_ >= np.pi/4 and crux_ < np.pi*3/4:
            return 1
        elif crux_ >= np.pi*3/4 and crux_ <= np.pi:
            return 3
        elif crux_ >= -np.pi/4 and crux_ < 0:
            return 0
        elif crux_ >= -np.pi*3/4 and crux_ < -np.pi/4:
            return 2
        elif crux_ > -np.pi and crux_ < -np.pi*3/4:
            return 4

    def rdp(self, traj):
        rdp_mask = rdp.rdp(traj, epsilon=1, return_mask=True, dist=point_to_segment_distance)
        # identify rapidly turn
        rdp_traj = traj[rdp_mask]
        rdp_diff = np.diff(rdp_traj, axis=0)
        rdp_rapid = (rdp_diff[:-1] * rdp_diff[1:]).sum(-1) / (np.linalg.norm(rdp_diff[:-1], axis=-1) * np.linalg.norm(rdp_diff[1:], axis=-1)) < -0.9
        rdp_rapid = np.concatenate([np.array([False]), rdp_rapid, np.array([False])])
        rapid_mask = np.zeros_like(rdp_mask)
        rapid_mask[rdp_mask] = rdp_rapid
        return rdp_mask, rapid_mask

    def rdp_traj(self, full_traj, color):
        traj_traj, traj_points, traj_idxs, traj_hit = self.radius_traj(full_traj, color)

        forward_idx = traj_idxs[traj_hit:]
        backward_idx = traj_idxs[:traj_hit+1][::-1]   # TODO remove +1
        rdp_mask, rapid_mask = self.rdp(traj_traj)
        forward_rapid = rapid_mask[traj_hit:]
        backward_rapid = rapid_mask[:traj_hit+1][::-1]   # TODO remove +1
        if forward_rapid.sum() > 0:
            forward_r = forward_idx[np.transpose(np.nonzero(forward_rapid))[0, 0]]
        else:
            forward_r = forward_idx[-1]
        if forward_rapid.sum() > 1:
            forward_r2 = forward_idx[np.transpose(np.nonzero(forward_rapid))[1, 0]]
        else:
            forward_r2 = forward_idx[-1]
        if backward_rapid.sum() > 0:
            backward_r = backward_idx[np.transpose(np.nonzero(backward_rapid))[0, 0]]
        else:
            backward_r = backward_idx[-1]
        if backward_rapid.sum() > 1:
            backward_r2 = backward_idx[np.transpose(np.nonzero(backward_rapid))[1, 0]]
        else:
            backward_r2 = backward_idx[-1]
        backward_color = backward_r
        forward_color = forward_r

        traj_idxs = np.arange(backward_color,forward_color+1)
        traj_traj = full_traj[traj_idxs]
        traj_points = self.convertor.car2matrix(traj_traj)
        traj_hit = color - backward_color

        return traj_traj, traj_points, traj_idxs, traj_hit

    def midpoint_control(self, points):
        return points[len(points) // 2]
    
    def perpendicular_control(self, points):
        a = points[0]
        b = points[-1]
        ab = (b - a)[None]  # 1 2
        ap = points - a[None]  # N 2

        if (a != b).sum() == 0:
            length = np.linalg.norm(ap, axis=-1)
        else:
            # Calculate the projection
            proj = ((ap * ab).sum(-1, keepdims=True) / (ab * ab).sum(-1, keepdims=True)) * ab
            # Calculate the vector component of a that is perpendicular
            perp = ap - proj
            # Calculate the magnitude of the perpendicular component
            length = np.linalg.norm(perp, axis=-1)
        
        # print("length", length)
        max_index = np.argmax(length)
        return points[max_index]

    def common_control(self, points_list):
        common_path = []
        for i in range(min([len(points) for points in points_list])):
            point = None
            for points in points_list:
                path = points[::-1]
                if point is None:
                    point = path[i]
                elif np.linalg.norm(np.array(path[i]) - np.array(point)) > 1e-6:
                    return common_path[-1]
            common_path.append(point)

    def traj_to_points_tangents(self, traj_points, traj_control):
        traj_tangents = traj_points[1:] - traj_points[:-1]
        traj_tangents_length = (traj_tangents[..., 0]**2 + traj_tangents[..., 1]**2) ** 0.5
        traj_tangents = np.where(traj_tangents_length[..., None] != 0, traj_tangents / traj_tangents_length[..., None], traj_tangents)

        curve_points, curve_tangents = curve_by_control(traj_points[0], traj_points[-1], traj_control)
        curve_points = np.array(curve_points, dtype=np.int32)
        curve_tangents = np.array(curve_tangents, dtype=np.float32)
        curve_tangents_length = np.linalg.norm(curve_tangents, axis=-1)
        curve_points = curve_points[curve_tangents_length != 0]
        curve_tangents = curve_tangents[curve_tangents_length != 0]

        traj_points = traj_points[:-1]

        return traj_points, traj_tangents, curve_points, curve_tangents

    def points_tangents_to_map(self, traj_points, traj_tangents, curve_points, curve_tangents, shape):

        traj_tangents_map, traj_distances_map = pierce_tangent(traj_points, traj_tangents, shape)
        curve_tangents_map, curve_distances_map = pierce_tangent(curve_points, curve_tangents, shape)
        traj_mask = (traj_points[:, 0] >= 0) & (traj_points[:, 0] < self.img_h) & (traj_points[:, 1] >= 0) & (traj_points[:, 1] < self.img_w)
        curve_mask = (curve_points[:, 0] >= 0) & (curve_points[:, 0] < self.img_h) & (curve_points[:, 1] >= 0) & (curve_points[:, 1] < self.img_w)
        traj_points_map = np.full((self.img_h, self.img_w), False)
        traj_points_map[traj_points[traj_mask, 0], traj_points[traj_mask, 1]] = True
        curve_points_map = np.full((self.img_h, self.img_w), False)
        curve_points_map[curve_points[curve_mask, 0], curve_points[curve_mask, 1]] = True

        # fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
        # ax = axes.ravel()
        # ax[0].scatter(traj_points[:, 0], traj_points[:, 1], s=0.1, c='red')
        # for i in range(traj_points.shape[0]):
        #     ax[0].annotate("{}".format(i), (traj_points[i, 0]+np.random.rand(), traj_points[i, 1]+np.random.rand()), fontsize=10)
        # ax[1].scatter(curve_points[:, 0], curve_points[:, 1], s=0.1, c='red')
        # for i in range(curve_points.shape[0]):
        #     ax[1].annotate("{} ({:.2f}, {:.2f})".format(i, curve_tangents[i, 0], curve_tangents[i, 1]), (curve_points[i, 0]+np.random.rand(), curve_points[i, 1]+np.random.rand()), fontsize=10)
        # plt.show()

        return traj_points_map, traj_tangents_map, traj_distances_map, curve_points_map, curve_tangents_map, curve_distances_map

    def convert_traj(self, traj_points, traj_control, shape):
        traj_points, traj_tangents, curve_points, curve_tangents = self.traj_to_points_tangents(traj_points, traj_control)
        traj_points_map, traj_tangents_map, traj_distances_map, curve_points_map, curve_tangents_map, curve_distances_map = self.points_tangents_to_map(traj_points, traj_tangents, curve_points, curve_tangents, shape)

        traj_tangents, traj_tangents_map = -traj_tangents, -traj_tangents_map
        curve_tangents, curve_tangents_map = -curve_tangents, -curve_tangents_map

        return traj_points_map, traj_tangents, traj_tangents_map, traj_distances_map, curve_points, curve_points_map, curve_tangents, curve_tangents_map, curve_distances_map

    def instruction2eline(self, ins_type):
        final_point = None
        if ins_type == 0:  # keep straight
            final_point = (0, self.img_w//2)
        elif ins_type == 1:  # turn left
            final_point = (self.img_h//2, 0)
        elif ins_type == 2:  # turn right
            final_point = (self.img_h//2, self.img_w-1)
        elif ins_type == 3:  # turnback left
            final_point = (self.img_h-1, 0)
        elif ins_type == 4:  # turnback right
            final_point = (self.img_h-1, self.img_w-1)
        else:
            raise NotImplementedError()
        traj_points = \
            np.array([
                (self.img_h-1, self.img_w//2),
                (self.img_h//2, self.img_w//2),
                final_point,
            ], dtype=np.int32)
        return traj_points

    def waypoint2eline(self, waypoint):
        ego = np.array([self.img_h//2, self.img_w//2])
        traj_points = \
            np.array([
                ego,
                (ego+waypoint) // 2,
                waypoint,
            ], dtype=np.int32)
        return traj_points

    def random_shift_rotate_indices_prepare(self, max_shift, max_angle):
        shift_y = np.random.uniform(-max_shift, max_shift)
        shift_x = np.random.uniform(-max_shift, max_shift)
        angle_rad = np.radians(np.random.uniform(-max_angle, max_angle))
        return shift_x, shift_y, angle_rad

    def random_shift_rotate_indices_convert(self, image_shape, shift_y, shift_x, angle_rad, flip_horizontal=False, flip_vertical=False):
        cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
        center_y, center_x = (np.array(image_shape) - 1) / 2.
        # Flip transformation matrix
        flip_x = -1 if flip_horizontal else 1
        flip_y = -1 if flip_vertical else 1
        # Combined transformation matrix: rotation + flip + translation
        # The order is: rotate around center, then flip around center, then translate
        transform_matrix = np.array([
            [flip_y * cos_angle, -flip_y * sin_angle, 
            -center_y * flip_y * cos_angle + center_x * flip_y * sin_angle + center_y + shift_y],
            [flip_x * sin_angle, flip_x * cos_angle, 
            -center_y * flip_x * sin_angle - center_x * flip_x * cos_angle + center_x + shift_x]
        ])
        return transform_matrix

    def random_shift_rotate_indices_apply(self, indices, transform_matrix):
        # Convert to homogeneous coordinates and apply transformation
        indices_homogeneous = np.column_stack([indices, np.ones(indices.shape[0])])
        rotated_indices = np.dot(transform_matrix, indices_homogeneous.T)
        rotated_indices = np.round(rotated_indices).astype(int)
        return rotated_indices.transpose()

    def instruction_eline(self, seq_id, frame_id, pose):
        full_correspondance = self.full_correspondances[seq_id]
        correspondance_id = full_correspondance[int(frame_id)]

        full_guidance = self.full_guidances[seq_id].copy()
        # full_guidance[:] += self.full_trajectorys[seq_id][int(frame_id)] - self.full_guidances[seq_id][correspondance_id]
        full_guidance = trans_by_pose2D(full_guidance, pose)
        full_guidance = trans_by_switch(full_guidance)
        _, guidance_points, guidance_idxs, guidance_hit = self.rdp_traj(full_guidance, correspondance_id)

        # instruction eline
        ins_type = self.rdp_instruction(guidance_points, guidance_hit)
        eline_points = self.instruction2eline(ins_type)
        eline_hit = 1
        return eline_points, eline_hit, full_guidance, guidance_idxs, ins_type

    def waypoint_eline(self, seq_id, frame_id, pose):
        full_correspondance = self.full_correspondances[seq_id]
        correspondance_id = full_correspondance[int(frame_id)]

        full_guidance = self.full_guidances[seq_id].copy()
        # full_guidance[:] += self.full_trajectorys[seq_id][int(frame_id)] - self.full_guidances[seq_id][correspondance_id]
        full_guidance = trans_by_pose2D(full_guidance, pose)
        full_guidance = trans_by_switch(full_guidance)
        _, guidance_points, guidance_idxs, guidance_hit = self.rdp_traj(full_guidance, correspondance_id)

        # waypoiny eline
        full_xmeter = self.full_xmeters[seq_id]
        xmeter = full_xmeter[guidance_idxs[-1]]
        waypoint_id = guidance_idxs[-1]
        waypoint_hit = len(guidance_points) - 1
        while waypoint_hit-1 > 0 and full_xmeter[waypoint_id-1] == xmeter:
            waypoint_id -= 1
            waypoint_hit -= 1
        eline_points = self.waypoint2eline(guidance_points[waypoint_hit])
        eline_hit = 1
        return eline_points, eline_hit, full_guidance, guidance_idxs

    def guidance_eline(self, seq_id, frame_id, pose):
        full_correspondance = self.full_correspondances[seq_id]
        correspondance_id = full_correspondance[int(frame_id)]

        full_guidance = self.full_guidances[seq_id].copy()
        # full_guidance[:] += self.full_trajectorys[seq_id][int(frame_id)] - self.full_guidances[seq_id][correspondance_id]
        full_guidance = trans_by_pose2D(full_guidance, pose)
        full_guidance = trans_by_switch(full_guidance)
        _, guidance_points, guidance_idxs, guidance_hit = self.rdp_traj(full_guidance, correspondance_id)
        # plt.scatter(guidance_points[:, 0], guidance_points[:, 1], s=0.1, c='red')
        # for i in range(guidance_points.shape[0]):
        #     plt.annotate("{}".format(guidance_idxs[i]), (guidance_points[i, 0]+np.random.rand(), guidance_points[i, 1]+np.random.rand()), fontsize=10)
        # plt.show()

        # guidance eline
        eline_points = guidance_points.copy()
        eline_hit = guidance_hit
        return eline_points, eline_hit, full_guidance, guidance_idxs

    def preprocess_to_bev(self, idx):
        seq_id, frame_id = self.dataset_parsePathInfoByIndex(idx)
        pose = self.dataset_poses[seq_id][int(frame_id)]

        single_acc_points, single_acc_sem_label, single_acc_inst_label = self.scans_single(idx)
        single_rgb_bev, single_label_bev = self.estimate_bev(single_acc_points, single_acc_sem_label, single_acc_inst_label)

        acc_points, acc_sem_label, acc_inst_label = self.scans_deploy(idx)
        rgb_bev, label_bev = self.estimate_bev(acc_points, acc_sem_label, acc_inst_label)

        if PREPROCESS_SCANS_TRAINING_ACC == 'temporal':
            full_acc_points, full_acc_sem_label, full_acc_inst_label = self.scans_training_temporal(idx)
            label_fusion = True
        elif PREPROCESS_SCANS_TRAINING_ACC == 'mapping':
            full_acc_points, full_acc_sem_label, full_acc_inst_label = self.scans_training_mapping(idx)
            label_fusion = False
        else:
            raise NotImplementedError()

        if PREPROCESS_SCANS_TRAINING_CROP == 'none':
            full_rgb_bev, full_label_bev = self.estimate_bev(full_acc_points, full_acc_sem_label, full_acc_inst_label, label_fusion=label_fusion)
        elif PREPROCESS_SCANS_TRAINING_CROP == 'height':
            trainingcop_full_acc_points, trainingcop_full_acc_sem_label, trainingcop_full_acc_inst_label = [], [], []
            for t1, t2, t3 in zip(full_acc_points, full_acc_sem_label, full_acc_inst_label):
                trainingcop_mask = t1[:, 2] + self.sensor_height <= 1
                tt1, tt2, tt3 = t1[trainingcop_mask], t2[trainingcop_mask], t3[trainingcop_mask]
                trainingcop_full_acc_points.append(tt1)
                trainingcop_full_acc_sem_label.append(tt2)
                trainingcop_full_acc_inst_label.append(tt3)
            full_rgb_bev, full_label_bev = self.estimate_bev(trainingcop_full_acc_points, trainingcop_full_acc_sem_label, trainingcop_full_acc_inst_label, label_fusion=label_fusion)
        else:
            raise NotImplementedError()

        single_rgb_bev_path = os.path.join(self.dataset_root, seq_id, "bev_rgb_single", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        single_label_bev_path = os.path.join(self.dataset_root, seq_id, "bev_label_single", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        rgb_bev_path = os.path.join(self.dataset_root, seq_id, "bev_rgb_post", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        label_bev_path = os.path.join(self.dataset_root, seq_id, "bev_label_post", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        full_rgb_bev_path = os.path.join(self.dataset_root, seq_id, "bev_rgb_all", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        full_label_bev_path = os.path.join(self.dataset_root, seq_id, "bev_label_all", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        if not os.path.exists(os.path.dirname(single_rgb_bev_path)):
            os.makedirs(os.path.dirname(single_rgb_bev_path))
        if not os.path.exists(os.path.dirname(single_label_bev_path)):
            os.makedirs(os.path.dirname(single_label_bev_path))
        if not os.path.exists(os.path.dirname(rgb_bev_path)):
            os.makedirs(os.path.dirname(rgb_bev_path))
        if not os.path.exists(os.path.dirname(label_bev_path)):
            os.makedirs(os.path.dirname(label_bev_path))
        if not os.path.exists(os.path.dirname(full_rgb_bev_path)):
            os.makedirs(os.path.dirname(full_rgb_bev_path))
        if not os.path.exists(os.path.dirname(full_label_bev_path)):
            os.makedirs(os.path.dirname(full_label_bev_path))
        with open(single_rgb_bev_path, 'wb') as f:
            np.save(f, single_rgb_bev)
        with open(single_label_bev_path, 'wb') as f:
            np.save(f, single_label_bev)
        with open(rgb_bev_path, 'wb') as f:
            np.save(f, rgb_bev)
        with open(label_bev_path, 'wb') as f:
            np.save(f, label_bev)
        with open(full_rgb_bev_path, 'wb') as f:
            np.save(f, full_rgb_bev)
        with open(full_label_bev_path, 'wb') as f:
            np.save(f, full_label_bev)

    def preprocess_from_bev(self, idx):
        seq_id, frame_id = self.dataset_parsePathInfoByIndex(idx)

        single_rgb_bev_path = os.path.join(self.dataset_root, seq_id, "bev_rgb_single", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        single_label_bev_path = os.path.join(self.dataset_root, seq_id, "bev_label_single", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        rgb_bev_path = os.path.join(self.dataset_root, seq_id, "bev_rgb_post", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        label_bev_path = os.path.join(self.dataset_root, seq_id, "bev_label_post", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        full_rgb_bev_path = os.path.join(self.dataset_root, seq_id, "bev_rgb_all", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        full_label_bev_path = os.path.join(self.dataset_root, seq_id, "bev_label_all", f"{frame_id}.npy").replace('semantic-kitti', "bev-kitti")
        with open(single_rgb_bev_path, 'rb') as f:
            single_rgb_bev = np.load(f)
        with open(single_label_bev_path, 'rb') as f:
            single_label_bev = np.load(f)
        with open(rgb_bev_path, 'rb') as f:
            rgb_bev = np.load(f)
        with open(label_bev_path, 'rb') as f:
            label_bev = np.load(f)
        with open(full_rgb_bev_path, 'rb') as f:
            full_rgb_bev = np.load(f)
        with open(full_label_bev_path, 'rb') as f:
            full_label_bev = np.load(f)

        return single_rgb_bev, single_label_bev, rgb_bev, label_bev, full_rgb_bev, full_label_bev

    def normalize_distance(self, distances):
        return normalize_distance(distances, self.img_h, self.img_w)

    def __getitem__(self, idx):
        timer = Timer(False)
        seq_id, frame_id = self.dataset_parsePathInfoByIndex(idx)
        pose = self.dataset_poses[seq_id][int(frame_id)]

        timer.put("forward, observation")
        if not self.is_deploy:
            rgb_bev, label_bev, full_rgb_bev, full_label_bev = self.training_data(idx)
        else:
            rgb_bev, label_bev, acc_points, acc_sem_label, acc_inst_label = self.deploy_data(idx)
        self.pad_bev_center(label_bev)
        label_erosion, label_padded, binary_traversable, label_traversable = self.estimate_traversable(label_bev)
        label_mask = label_padded == self.dataset_label_traversable
        timer.get()

        timer.put("guidance")
        if not self.is_deploy:
            self.pad_bev_center(full_label_bev)
            full_label_erosion, full_label_padded, full_binary_traversable, full_label_traversable = self.estimate_traversable(full_label_bev)
            # full_rgb_bev, full_label_bev, full_label_padded, full_binary_traversable, full_label_traversable = rgb_bev, label_bev, label_padded, binary_traversable, label_traversable
            full_label_mask = full_label_padded == self.dataset_label_traversable

            full_binary_frontier, full_cluster_frontier, full_label_pollute, full_baised_skeleton, full_baised_frontier_skeleton, full_skeleton_binary_traversable, full_skeleton_cluster_frontier = \
                estimate_frontier_and_skeleton(full_label_padded, full_binary_traversable)
            panel_binary_traversable, panel_cluster_frontier, panel_biased_frontier = full_binary_traversable, full_skeleton_cluster_frontier, full_baised_frontier_skeleton
            road_end_label, road_end_cnt = np.unique(panel_cluster_frontier, return_counts=True)
            road_end = road_end_label[1:]
            all_end = panel_cluster_frontier
            to_end = panel_biased_frontier

            raw_eline_points = np.full((0, 2), 0)
            eline_points = np.full((0, 2), 0)
            eline_control = None
            if len(road_end) >= 2:
                to_boundary = all_end == to_end
                from_ends = road_end[road_end != to_end]

                dij = planning.Dijkstra(panel_binary_traversable)
                movements, tangents = dij.plan(to_boundary)
                shift_x, shift_y, angle_rad = self.random_shift_rotate_indices_prepare(min(self.img_h, self.img_w) // 4, 0)
                # shift_x, shift_y, angle_rad = 0, 0, 0
                noising_transform_matrix = self.random_shift_rotate_indices_convert(label_padded.shape, shift_x, shift_y, angle_rad)
                for i, from_end in enumerate(from_ends[np.random.permutation(len(from_ends))]):
                    from_boundary = all_end == from_end
                    from_idxs = np.transpose(np.nonzero(from_boundary))
                    current = tuple(from_idxs[cdist([from_idxs.mean(axis=0)], from_idxs).argmin()])
                    path_points, path_tangents = dij.back_tracing(movements, tangents, current)
                    tmp_raw_eline_points = path_points[::-1]
                    tmp_eline_points = self.random_shift_rotate_indices_apply(tmp_raw_eline_points, noising_transform_matrix)
                    if path_points.shape[0] > min(self.img_h, self.img_w) // 2:  # 至少大于短边的一半
                        raw_eline_points = tmp_raw_eline_points
                        eline_points = tmp_eline_points
                        break
                if len(eline_points) > 0:
                    to_boundary_traversable = full_cluster_frontier == to_end
                    dists, _, _, _, d_tangents, _, inverse_grads, _, combines = \
                        edt_with_tangent(full_binary_traversable, to_boundary_traversable)#, invalid=full_label_padded == 0
                    edt_distances = dists.copy()
                    edt_tangents = combines.copy()
                    edt_mask = ~to_boundary_traversable

                    shift_x, shift_y, angle_rad = 0, 0, 0
                    flip_horizontal, flip_vertical = False, False
                    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                    if enable:
                        # enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                        # if enable:
                        #     shift_x = np.random.uniform(-self.img_h // 4, self.img_h // 4)
                        #     shift_y = np.random.uniform(-self.img_w // 4, self.img_w // 4)
                        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                        if enable:
                            angle_rad = np.random.uniform(0, np.pi*2)
                        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                        if enable:
                            flip_horizontal = True
                        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
                        if enable:
                            bev_drop = np.mgrid[:label_padded.shape[0], :label_padded.shape[1]].reshape(2, -1).T[np.random.permutation(label_padded.shape[0]*label_padded.shape[1])][:int(np.random.randint(0, label_padded.shape[0]*label_padded.shape[1]) * 0.25)]
                            rgb_bev[bev_drop.T[0], bev_drop.T[1]] = 0
                    augmentation_transform_matrix = self.random_shift_rotate_indices_convert(label_padded.shape, shift_x, shift_y, angle_rad, flip_horizontal, flip_vertical)

                    raw_eline_points = (augmentation_transform_matrix @ np.column_stack([raw_eline_points, np.ones(raw_eline_points.shape[0])]).T).T.astype(np.int32)
                    eline_points = (augmentation_transform_matrix @ np.column_stack([eline_points, np.ones(eline_points.shape[0])]).T).T.astype(np.int32)

                    edt_points = np.mgrid[:label_padded.shape[0], :label_padded.shape[1]].reshape(2, -1).T
                    # Apply homogeneous transformation
                    homogeneous_edt_points = np.column_stack([edt_points, np.ones(edt_points.shape[0])])
                    transformed_edt_points = (augmentation_transform_matrix @ homogeneous_edt_points.T).T
                    transformed_edt_distances = edt_distances.reshape(-1)
                    transformed_edt_tangents = (augmentation_transform_matrix[:2, :2] @ edt_tangents.reshape(-1, 2).T).T
                    transformed_edt_mask = edt_mask.reshape(-1)

                    transformed_edt_points = transformed_edt_points.astype(np.int32)
                    transformed_edt_points_mask = (transformed_edt_points[:, 0] >= 0) & (transformed_edt_points[:, 0] < label_padded.shape[0]) & (transformed_edt_points[:, 1] >= 0) & (transformed_edt_points[:, 1] < label_padded.shape[1])
                    edt_distances = np.full(edt_distances.shape, 0.)
                    edt_distances[transformed_edt_points.T[0, transformed_edt_points_mask], transformed_edt_points.T[1, transformed_edt_points_mask]] = transformed_edt_distances[transformed_edt_points_mask]
                    edt_tangents = np.full(edt_tangents.shape, 0.)
                    edt_tangents[transformed_edt_points.T[0, transformed_edt_points_mask], transformed_edt_points.T[1, transformed_edt_points_mask]] = transformed_edt_tangents[transformed_edt_points_mask]
                    edt_mask = np.full(edt_mask.shape, False)
                    edt_mask[transformed_edt_points.T[0, transformed_edt_points_mask], transformed_edt_points.T[1, transformed_edt_points_mask]] = transformed_edt_mask[transformed_edt_points_mask]

                    transformed_rgb_bev = rgb_bev.reshape(-1, 3)
                    rgb_bev = np.full(rgb_bev.shape, 0.)
                    rgb_bev[transformed_edt_points.T[0, transformed_edt_points_mask], transformed_edt_points.T[1, transformed_edt_points_mask]] = transformed_rgb_bev[transformed_edt_points_mask]

                    eline_control = self.perpendicular_control(eline_points)
        else:
            full_acc_points, full_acc_sem_label, full_acc_inst_label = self.scans_training_temporal(idx)
            full_rgb_bev, full_label_bev = self.estimate_bev(full_acc_points, full_acc_sem_label, full_acc_inst_label)
            full_label_erosion, full_label_padded, full_binary_traversable, full_label_traversable = self.estimate_traversable(full_label_bev)
            full_label_mask = full_label_padded == self.dataset_label_traversable

            eline_points, eline_hit, full_guidance, guidance_idxs, ins_type = self.instruction_eline(seq_id, frame_id, pose)
            eline_points, eline_hit, full_guidance, guidance_idxs = self.guidance_eline(seq_id, frame_id, pose)
            eline_control = self.perpendicular_control(eline_points)
            eline_traj = full_guidance[guidance_idxs]
        timer.get()

        timer.put("guidance")
        eline_points_map = np.full((self.img_h, self.img_w), False)
        eline_tangents = np.full((0, 2)+(2,), 0.)
        eline_tangents_map = np.full((self.img_h, self.img_w)+(2,), 0.)
        curve_points = np.full((0, 2), 0)
        curve_points_map = np.full((self.img_h, self.img_w), False)
        curve_tangents = np.full((0, 2)+(2,), 0.)
        curve_tangents_map = np.full((self.img_h, self.img_w)+(2,), 0.)
        curve_distances_map = np.full((self.img_h, self.img_w), 0.)
        if len(eline_points) > 0:
            eline_points_map, eline_tangents, eline_tangents_map, eline_distances_map, \
                curve_points, curve_points_map, curve_tangents, curve_tangents_map, curve_distances_map = \
                    self.convert_traj(eline_points, eline_control, label_padded.shape)
        timer.get()

        timer.put("guidance2")
        greedy_point = np.full((2), 0)
        greedy_points = np.full((0, 2), 0)
        greedy_tangents = np.full((0, 2)+(2,), 0.)
        if len(eline_points) > 0:
            greedy_point = eline_points[-1]
            greedy_points = eline_points.copy()
            greedy_tangents = eline_tangents.copy()
        timer.get()

        timer.put("forward")
        tangent_init = curve_tangents_map.copy()
        distance_init = self.normalize_distance(curve_distances_map)
        tangent_raw = np.full(label_padded.shape+(2,), 0.)
        tangent_map = np.full(label_padded.shape+(2,), 0.)
        distance_map = np.full(label_padded.shape, 0.)
        training_mask = np.full(label_padded.shape, False)
        if not self.is_deploy and len(eline_points) > 0:
            tangent_map = edt_tangents.copy()
            distance_map = 1 - self.normalize_distance(edt_distances)
            training_mask = edt_mask.copy()
        timer.get()

        timer.put("target")
        if not self.is_deploy:
            raw_eline_points_mask = (raw_eline_points[:, 0] >= 0) & (raw_eline_points[:, 0] < label_padded.shape[0]) & (raw_eline_points[:, 1] >= 0) & (raw_eline_points[:, 1] < label_padded.shape[1])
            target_points = raw_eline_points[raw_eline_points_mask]
            target_traj = self.convertor.matrix2car(target_points)
            target_points_past = target_points[:0]
            target_traj_past = target_traj[:0]
        else:
            full_trajectory = self.full_trajectorys[seq_id].copy()
            full_trajectory = trans_by_pose2D(full_trajectory, pose)
            full_trajectory = trans_by_switch(full_trajectory)
            trajectory_traj, trajectory_points, trajectory_idxs, trajectory_hit = self.cut_traj(full_trajectory, int(frame_id))
            trajectory_traj_rdp_mask, trajectory_traj_rapid_mask = self.rdp(trajectory_traj)
            trajectory_traj_idxs_forward = trajectory_idxs[trajectory_hit:]
            trajectory_traj_idxs_backward = trajectory_idxs[:trajectory_hit][::-1]
            trajectory_traj_rapid_forward = trajectory_traj_rapid_mask[trajectory_hit:]
            trajectory_traj_rapid_backward = trajectory_traj_rapid_mask[:trajectory_hit][::-1]
            if trajectory_traj_rapid_forward.sum() > 0:
                trajectory_traj_forward_r = np.arange(np.transpose(np.nonzero(trajectory_traj_rapid_forward))[0, 0]+1)
            else:
                trajectory_traj_forward_r = np.arange(trajectory_traj_rapid_forward.shape[0])
            if trajectory_traj_rapid_backward.sum() > 0:
                trajectory_traj_backward_r = np.arange(np.transpose(np.nonzero(trajectory_traj_rapid_backward))[0, 0]+1)
                # use backward rapid to limit forward distance
                trajectory_traj_backward_distance = np.cumsum(np.linalg.norm(np.diff(np.concatenate([trajectory_traj[trajectory_hit:trajectory_hit+1], trajectory_traj[:trajectory_hit][::-1][trajectory_traj_backward_r]], axis=0), axis=0), axis=1))[-1]
                trajectory_traj_forward_distance = np.concatenate([np.array([0]), np.cumsum(np.linalg.norm(np.diff(trajectory_traj[trajectory_hit:][trajectory_traj_forward_r], axis=0), axis=1))])
                if (trajectory_traj_forward_distance > trajectory_traj_backward_distance).sum() > 0:
                    trajectory_traj_forward_r = trajectory_traj_forward_r[:np.transpose(np.nonzero(trajectory_traj_forward_distance > trajectory_traj_backward_distance))[0, 0]]
            else:
                trajectory_traj_backward_r = np.arange(trajectory_traj_rapid_backward.shape[0])
            trajectory_traj_backward_forward_r = np.concatenate([trajectory_traj_idxs_backward[trajectory_traj_backward_r][::-1], trajectory_traj_idxs_forward[trajectory_traj_forward_r]], axis=0) - trajectory_idxs[0]
            trajectory_traj, trajectory_points, trajectory_idxs = trajectory_traj[trajectory_traj_backward_forward_r], trajectory_points[trajectory_traj_backward_forward_r], trajectory_idxs[trajectory_traj_backward_forward_r]
            trajectory_hit -= trajectory_traj_backward_forward_r[0]
            target_points = trajectory_points[trajectory_hit:]
            target_traj = trajectory_traj[trajectory_hit:]
            target_points_past = trajectory_points[:trajectory_hit]
            target_traj_past = trajectory_traj[:trajectory_hit]
        target_points_map = np.full((self.img_h, self.img_w), False)
        target_points_map[target_points[:, 0], target_points[:, 1]] = True
        target_points_past_map = np.full((self.img_h, self.img_w), False)
        target_points_past_map[target_points_past[:, 0], target_points_past[:, 1]] = True
        timer.get()

        timer.flush()

        data_dict = {
            # forward
            'bev_map': rgb_bev,
            'tangent_init': tangent_init,
            'distance_init': distance_init,
            'tangent_raw': tangent_raw,
            'tangent_map': tangent_map,
            'distance_map': distance_map,
            'training_mask': training_mask,

            # observation
            'bev_label': label_bev,
            'label_mask': label_mask,
            'label_padded': label_padded,
            'binary_traversable': binary_traversable,
            "full_bev_label": full_label_bev,
            'full_label_mask': full_label_mask,
            "full_label_padded": full_label_padded,
            "full_binary_traversable": full_binary_traversable,

            # guidance
            'eline_points': eline_points,
            'eline_points_map': eline_points_map,
            'eline_tangents': eline_tangents,
            'eline_tangents_map': eline_tangents_map,
            'curve_points': curve_points,
            'curve_points_map': curve_points_map,
            'curve_tangents': curve_tangents,
            'curve_tangents_map': curve_tangents_map,

            # guidance2
            'greedy_point': greedy_point,
            'greedy_points': greedy_points,
            'greedy_tangents': greedy_tangents,

            # target
            'target_points': target_points,
            "target_points_map": target_points_map,
            'target_traj': target_traj,
            'target_points_past': target_points_past,
            "target_points_past_map": target_points_past_map,
            'target_traj_past': target_traj_past,
        }

        if self.is_deploy:
            acc_points, acc_sem_label, acc_inst_label = np.concatenate(acc_points, axis=0), np.concatenate(acc_sem_label, axis=0), np.concatenate(acc_inst_label, axis=0)
            # for baseline inference
            data_dict['pointcloud'] = acc_points
            data_dict['sem_label'] = acc_sem_label
            data_dict['inst_label'] = acc_inst_label
            # for temporal aggregation
            data_dict['pose'] = pose
            # for preprocess
            data_dict['eline_hit'] = eline_hit
            data_dict['eline_traj'] = eline_traj
            data_dict['ins_type'] = ins_type

        if LOADER_VIS:
            figure = plt.figure(figsize=(30, 15))
            axesimg = figure.add_subplot(3, 8, 1)
            axesimg.imshow(self.visulizater.naive_vis(rgb_bev))
            axesimg.axis("off")
            axesimg = figure.add_subplot(3, 8, 2)
            axesimg.imshow(self.visulizater.stack_layer(
                    self.visulizater.tangent_vis(tangent_init, pltcm='hsv'), 
                    [eline_points_map, curve_points_map] + [self.convertor.matrix_point2map(eline_points[0], radius=4), self.convertor.matrix_point2map(eline_control, radius=4), self.convertor.matrix_point2map(eline_points[-1], radius=4)] if len(eline_points) > 0 else [], 
                    fg_color_list=[np.array([0,0,0,1.]),np.array([0,0,0,1.])] + [np.array([1.,1.,1.,1.]), np.array([1.,1.,1.,1.]), np.array([1.,1.,1.,1.])] if len(eline_points) > 0 else []))
            axesimg.axis("off")
            axesimg = figure.add_subplot(3, 8, 3)
            axesimg.imshow(self.visulizater.tangent_vis(tangent_init, pltcm='hsv'))
            axesimg.axis("off")
            axesimg = figure.add_subplot(3, 8, 4)
            colorbase = np.full((self.img_h, self.img_w, 4), 1.)
            colorbase[..., :3] = self.visulizater.naive_vis(rgb_bev)
            axesimg.imshow(self.visulizater.merge_layer(
                    self.visulizater.tangent_vis(tangent_init, pltcm='hsv'), 
                    colorbase,
                    label_bev > 0))
            axesimg.axis("off")
            axesimg = figure.add_subplot(3, 8, 9)
            axesimg.imshow(distance_init)
            axesimg.axis("off")
            if not self.is_deploy:
                axesimg = figure.add_subplot(3, 2, 2)
                axesimg.imshow(np.concatenate([
                    self.visulizater.defined_label_vis(full_label_bev),
                    self.visulizater.defined_label_vis(full_label_padded),
                    self.visulizater.binary_label_vis(full_binary_traversable),
                    self.visulizater.clustered_label_vis(full_cluster_frontier),
                ], axis=1))
                axesimg.axis("off")
                if len(eline_points) > 0:
                    axesimg = figure.add_subplot(3, 2, 4)
                    colorbase = np.full((self.img_h, self.img_w, 4), 1.)
                    colorbase[..., :3] = self.visulizater.naive_vis(full_rgb_bev)
                    colorbase2 = np.full((self.img_h, self.img_w, 4), 1.)
                    colorbase2[..., :3] = self.visulizater.stack_layer(
                        self.visulizater.clustered_label_vis(full_cluster_frontier),
                        [full_binary_traversable & (full_cluster_frontier == 0), target_points_map, eline_points_map], 
                        fg_color_list=[np.array([1.,1.,1.]), np.array([0.,0.,0.]), np.array([1.,0.,0.])])
                    axesimg.imshow(np.concatenate([
                        colorbase,
                        colorbase2,
                        self.visulizater.tangent_vis(tangents, tangent_mask=full_binary_traversable, pltcm='hsv'),
                        self.visulizater.tangent_vis(tangent_map, tangent_mask=training_mask, pltcm='hsv'),
                    ], axis=1))
                    axesimg.axis("off")
                    axesimg = figure.add_subplot(3, 8, 21)
                    axesimg.imshow(distance_map)
                    axesimg.axis("off")
            else:
                axescurrent = figure.add_subplot(1, 2, 2)
                axescurrent.set_aspect('equal')
                full_correspondance = full_guidance[self.full_correspondances[seq_id]]
                axescurrent.scatter(full_trajectory[:, 0], full_trajectory[:, 1], s=0.1, c='green')
                axescurrent.scatter(full_guidance[:, 0], full_guidance[:, 1], s=0.1, c='red')
            figure.suptitle(f"seq {seq_id} frame {frame_id}")
            cmap = plt.cm.get_cmap('hsv')
            display_axes = figure.add_axes([0.95, 0.95, 0.05, 0.05], projection='polar')
            bar = mpl.colorbar.ColorbarBase(display_axes, cmap=cmap,
                                    norm=mpl.colors.Normalize(0.0, 2*np.pi),
                                    orientation='horizontal')
            bar.outline.set_visible(False)
            display_axes.set_axis_off()
            display_axes.set_theta_offset(-np.pi/2)
            plt.tight_layout()
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.show()

        return data_dict


DATASET_PATH = os.getenv('DATASET_PATH')
DATASET_TYPE = os.getenv('DATASET_TYPE')
assert DATASET_TYPE in ["raw-derived-semantic-kitti", "semantic-kitti", "um-osm-planner"]
# switch dataset should modify 1 place in evaluator if deploy


class NGMLoader(NGMDataset, NGMInterfance):
    def __init__(self, dataset, logger=None, is_deploy=False):
        self.dataset = dataset
        self.is_deploy = is_deploy

        if DATASET_TYPE == "raw-derived-semantic-kitti" or "semantic-kitti":
            self.dataset_scans_training_n = 6
            self.dataset_scans_deploy_n = 3
            self.dataset_label_traversable = 40
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bev-semantic-kitti.yaml")
        elif DATASET_TYPE == "um-osm-planner":
            self.dataset_scans_training_n = -1
            self.dataset_scans_deploy_n = 20
            self.dataset_label_traversable = 3
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bev-um-osmplanner.yaml")
        else:
            raise NotImplementedError()

        with open(cfg_path) as fp:
            if hasattr(yaml, 'FullLoader'):
                config_dict = yaml.load(fp, Loader=yaml.FullLoader)
            else:
                config_dict = yaml.load(fp)
        super(NGMLoader, self).__init__(cfg=merge_new_config(Dict(), Dict(config_dict)), logger=logger)
    
    def training_data(self, idx):
        if DATASET_TYPE == "raw-derived-semantic-kitti" or "semantic-kitti":
            rgb_bev, label_bev, _, _, full_rgb_bev, full_label_bev = self.preprocess_from_bev(idx)
        elif DATASET_TYPE == "um-osm-planner":
            _, _, rgb_bev, label_bev, full_rgb_bev, full_label_bev = self.preprocess_from_bev(idx)
        else:
            raise NotImplementedError()
        return rgb_bev, label_bev, full_rgb_bev, full_label_bev

    def deploy_data(self, idx):
        if DATASET_TYPE == "raw-derived-semantic-kitti" or "semantic-kitti":
            acc_points, acc_sem_label, acc_inst_label = self.scans_single(idx)
        elif DATASET_TYPE == "um-osm-planner":
            acc_points, acc_sem_label, acc_inst_label = self.scans_deploy(idx)
        else:
            raise NotImplementedError()
        rgb_bev, label_bev = self.estimate_bev(acc_points, acc_sem_label, acc_inst_label)
        return rgb_bev, label_bev, acc_points, acc_sem_label, acc_inst_label


if DATASET_TYPE == "raw-derived-semantic-kitti":
    from api.datasetapi.dataset import RawDerivedSemanticKitti
    data_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic-kitti.yaml")
    trainset = RawDerivedSemanticKitti(
     panoptic=False,
     root=DATASET_PATH,
     sequences=[0, 2, 5, 7],
     config_path=data_config_path,
     has_image=False
    )
    valset = RawDerivedSemanticKitti(
     panoptic=False,
     root=DATASET_PATH,
     sequences=DEPLOY_SEQS.split(",") if DEPLOY_SEQS is not None else [10],
     config_path=data_config_path,
     has_image=False
    )
elif DATASET_TYPE == "semantic-kitti":
    from api.datasetapi.dataset import SemanticKitti
    data_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "semantic-kitti.yaml")
    trainset = SemanticKitti(
        panoptic=False,
        root=DATASET_PATH,
        sequences=[0, 1, 2, 3, 4, 5, 6, 7, 9, 10],
        config_path=data_config_path
    )
    valset = SemanticKitti(
        panoptic=False,
        root=DATASET_PATH,
        sequences=DEPLOY_SEQS.split(",") if DEPLOY_SEQS is not None else [8],
        config_path=data_config_path
    )
elif DATASET_TYPE == "um-osm-planner":
    from api.datasetapi.dataset import UMOSMPlanner
    trainset = UMOSMPlanner(
        panoptic=False,
        root=DATASET_PATH,
        splits=["scene 3","scene 4"],
        mode='livox',
    )
    valset = UMOSMPlanner(
        panoptic=False,
        root=DATASET_PATH,
        splits=DEPLOY_SEQS.split(",") if DEPLOY_SEQS is not None else ["scene mini"],
        mode='livox',
    )
else:
    raise NotImplementedError()


logger = logging.getLogger(__name__)
trainset_loader = NGMLoader(trainset, logger, is_deploy=False)
valset_loader = NGMLoader(valset, logger, is_deploy=DEPLOY_SEQS is not None)