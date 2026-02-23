import os
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from skimage.measure import block_reduce
from skimage.draw import line

from lidardet.ops.planning import planning

from .ngm import smooth_and_brocast_tangent


def map_traj(rrt_traj, img_h, img_w, rrt_tree_map=None):
    rrt_planned_map = np.full((img_h, img_w), 0.)
    for i in range(1, len(rrt_traj)):
        current_node = tuple(rrt_traj[i-1])
        child_node = tuple(rrt_traj[i])
        current_node_uv = (current_node[1], current_node[0])
        child_node_uv = (child_node[1], child_node[0])
        cv2.line(rrt_planned_map, current_node_uv, child_node_uv, (0.5, 0.5, 0.5), 1)
        if rrt_tree_map is not None:
            rrt_planned_map[current_node] = rrt_tree_map[current_node]
            rrt_planned_map[child_node] = rrt_tree_map[child_node]
    rrt_planned_map = np.where(rrt_planned_map == 0.5, np.ones_like(rrt_planned_map) * 3, rrt_planned_map)
    return rrt_planned_map


class Dijkstra:
    def __init__(self, traversable_map):
        # Define the cost to move to each direction
        d1, d2 = 1, np.sqrt(2)
        self.directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]  # 8-connectivity
        self.costs = [d1, d2, d1, d2, d1, d2, d1, d2]

        self.traversable_map = traversable_map

    def plan(self, start_map):
        traversability_map = np.zeros(self.traversable_map.shape, dtype=np.int32)  # 0: dont visit, 1: to visit, 2: visited
        traversability_map[self.traversable_map != 0] = 1
        traversability_map[start_map] = 2
        movements = planning.dijkstra(traversability_map, np.transpose(np.nonzero(start_map)), self.directions, self.costs)
        tangents = self.tangent(movements)
        return movements, tangents

    def randomwalk(self, start_map):
        traversability_map = np.zeros(self.traversable_map.shape, dtype=np.int32)  # 0: dont visit, 1: to visit, 2: visited
        traversability_map[self.traversable_map != 0] = 1
        traversability_map[start_map] = 2
        movements = planning.randomwalk(traversability_map, np.transpose(np.nonzero(start_map)), self.directions, self.costs)
        tangents = self.tangent(movements)
        return movements, tangents

    def back_trace(self, movements, idx):
        from_ = self.directions[movements[idx]-1]
        return (idx[0]-from_[0], idx[1]-from_[1])

    def back_tracing(self, movements, tangents, current):
        current = tuple(current)
        trace_points = []
        trace_tangents = []
        while movements[current] != 0:
            trace_points.append(list(current))
            trace_tangents.append(tangents[current])
            current = self.back_trace(movements, current)
        trace_points = np.array(trace_points)
        trace_tangents = np.array(trace_tangents)
        if trace_points.shape[0] > 0:
            return trace_points[::-1], trace_tangents[::-1]
        else:
            return np.full((0, 2), 0), np.full((0, 2, 2), 0.)

    def tangent(self, movements):
        directions = np.array([list(item) for item in self.directions], dtype=np.int32)
        tangents = directions[movements-1].astype(np.float32)
        tangents_length = (tangents[..., 0]**2 + tangents[..., 1]**2) ** 0.5
        tangents = np.where(tangents_length[..., None] != 0, tangents / tangents_length[..., None], tangents)
        return np.where((movements != 0)[..., None], tangents, np.zeros_like(tangents))


class RRT:
    def __init__(self, extend_mode, extend_area, traversability_map, conf_map, tangent_map, tangent_map2, discorage_map):
        self.stepSize = 0.05
        self.maxIter = 1000
        self.neighborRadius = 0.1
        self.neighborCount = -1
        self.extend_mode = extend_mode
        self.extend_area = extend_area

        self.traversability_map = traversability_map
        self.conf_map = conf_map
        self.tangent_map = tangent_map
        self.tangent_map2 = tangent_map2
        self.discorage_map = discorage_map

    def plan(self, res, key_meters, greedy_points=None, start=None, pro_points=None):
        img_h, img_w = self.traversability_map.shape

        if start is None:
            start = (int(img_h/2), int(img_w/2))
        if pro_points is None:
            pro_points = np.full((0, 2), 0)
        key_pixels = int(key_meters / res)
        canvas = np.zeros((img_h, img_w), dtype=np.uint8)
        color = 255        # Color of the circle (white)
        thickness = 1       # Thickness (-1 will fill the circle)
        cv2.circle(canvas, (start[1], start[0]), key_pixels, color, thickness)
        key_map = canvas == color

        sampleRadius = key_pixels / img_w
        if self.extend_area == 1:
            extend_map = self.traversability_map.copy()
        elif self.extend_area == 2:
            extend_map = np.ones_like(self.traversability_map)
        else:
            raise NotImplementedError()
        tangent_map_x = self.tangent_map[..., 0]
        tangent_map_y = self.tangent_map[..., 1]
        tangent_map2_x = self.tangent_map2[..., 0]
        tangent_map2_y = self.tangent_map2[..., 1]
        rrt_scale = sampleRadius * 2
        rrt_map = planning.rrt(extend_map, self.conf_map, tangent_map_x, tangent_map_y, tangent_map2_x, tangent_map2_y, self.discorage_map, start, pro_points, np.transpose(np.nonzero(key_map)), 
                               sampleRadius, self.stepSize * rrt_scale, self.maxIter, self.neighborRadius * rrt_scale, self.neighborCount, self.extend_mode)
        rrt_map = rrt_map.reshape((img_h, img_w, -1)).copy()

        rrt_tree_map = rrt_map[..., 0].astype(np.int32)
        rrt_parentx_map = rrt_map[..., 1].astype(np.int32)
        rrt_parenty_map = rrt_map[..., 2].astype(np.int32)
        rrt_cost_map = rrt_map[..., 3]
        rrt_energy_map = rrt_map[..., 4]
        rrt_step_map = rrt_map[..., 5].astype(np.int32)
        rrt_inc_cost_map = rrt_map[..., 6]
        rrt_inc_energy_map = rrt_map[..., 7]
        rrt_inc_gain_map = rrt_inc_energy_map
        rrt_gain_map = 0.5 * rrt_cost_map + rrt_energy_map
        rrt_consume_map = 0.5 * rrt_cost_map - rrt_energy_map

        if self.extend_area == 1:
            candidate_points_map = (rrt_tree_map == 1) | (rrt_tree_map == 2)
        elif self.extend_area == 2:
            candidate_points_map = ((rrt_tree_map == 1) | (rrt_tree_map == 2)) & key_map
        else:
            raise NotImplementedError()
        candidate_points = np.transpose(np.nonzero(candidate_points_map))
        if candidate_points.shape[0] == 0:
            current_node = None
        else:
            if greedy_points is None:
                current_node = tuple(candidate_points[rrt_consume_map[candidate_points.T[0], candidate_points.T[1]].argmin()])
            else:
                if greedy_points.shape[0] == 0:
                    current_node = None
                else:
                    dist_candidate_greedy = np.linalg.norm(candidate_points[:, None, :] - greedy_points[None, :, :], axis=-1)
                    dist_candidate = dist_candidate_greedy.min(-1)
                    current_node = tuple(candidate_points[dist_candidate.argmin()])
        # from matplotlib import pyplot as plt
        # plt.imshow(rrt_tree_map)
        # plt.show()
        rrt_traj = np.full((0, 2), 0)
        while current_node is not None:
            # print("current_node", current_node)
            rrt_traj = np.append(rrt_traj, np.array([list(current_node)]), axis=0)
            current_node = (rrt_parentx_map[current_node], rrt_parenty_map[current_node])
            if current_node[0] == -1 and current_node[1] == -1:
                break
        rrt_traj = rrt_traj[::-1]
        rrt_planned_map = map_traj(rrt_traj, img_h, img_w, rrt_tree_map)
        # print("rrt_traj")

        return rrt_tree_map, rrt_gain_map, rrt_inc_energy_map, rrt_inc_gain_map, rrt_planned_map, rrt_traj


class ScaleRRT(RRT):
    def __init__(self, extend_mode, extend_area, traversability_map, conf_map, tangent_map, tangent_map2, discorage_map, scale=2):
        super(ScaleRRT, self).__init__(extend_mode, extend_area, traversability_map, conf_map, tangent_map, tangent_map2, discorage_map)

        self.scale = scale
        x, y = np.mgrid[:self.traversability_map.shape[0], :self.traversability_map.shape[1]]
        self.scale_mask = (x % self.scale == 0) & (y % self.scale == 0)
        self.init_h, self.init_w = self.traversability_map.shape
        self.traversability_map = block_reduce(self.traversability_map, (self.scale, self.scale), np.mean) > 0.5
        self.conf_map = block_reduce(self.conf_map, (self.scale, self.scale), np.mean)
        self.tangent_map = block_reduce(self.tangent_map, (self.scale, self.scale, 1), np.mean)
        self.tangent_map2 = block_reduce(self.tangent_map2, (self.scale, self.scale, 1), np.mean)
        self.discorage_map = block_reduce(self.discorage_map, (self.scale, self.scale), np.mean)
        # print("self.traversability_map", self.traversability_map)
        # print("self.conf_map", self.conf_map)
        # print("self.tangent_map", self.tangent_map)
        # print("self.discorage_map", self.discorage_map)

    def plan(self, res, key_meters, greedy_points=None, start=None, pro_points=None):

        scaled_rrt_tree_map, scaled_rrt_gain_map, scaled_rrt_inc_energy_map, scaled_rrt_inc_gain_map, scaled_rrt_planned_map, scaled_rrt_traj = \
            super(ScaleRRT, self).plan(res * self.scale, key_meters, 
                                       (greedy_points // self.scale).astype(greedy_points.dtype) if greedy_points is not None else greedy_points, 
                                       (start // self.scale).astype(start.dtype) if start is not None else start,
                                       (pro_points // self.scale).astype(pro_points.dtype) if pro_points is not None else pro_points,
                                       )

        # from matplotlib import pyplot as plt
        # plt.imshow(scaled_rrt_tree_map)
        # plt.show()
        rrt_tree_map = np.zeros((self.init_h, self.init_w), dtype=scaled_rrt_tree_map.dtype)
        rrt_gain_map = np.zeros((self.init_h, self.init_w), dtype=scaled_rrt_gain_map.dtype)
        rrt_inc_energy_map = np.zeros((self.init_h, self.init_w), dtype=scaled_rrt_inc_energy_map.dtype)
        rrt_inc_gain_map = np.zeros((self.init_h, self.init_w), dtype=scaled_rrt_inc_gain_map.dtype)
        rrt_planned_map = np.zeros((self.init_h, self.init_w), dtype=scaled_rrt_planned_map.dtype)
        rrt_tree_map[self.scale_mask] = scaled_rrt_tree_map.flatten()
        rrt_gain_map[self.scale_mask] = scaled_rrt_gain_map.flatten()
        rrt_inc_energy_map[self.scale_mask] = scaled_rrt_inc_energy_map.flatten()
        rrt_inc_gain_map[self.scale_mask] = scaled_rrt_inc_gain_map.flatten()
        rrt_planned_map[self.scale_mask] = scaled_rrt_planned_map.flatten()
        rrt_traj = scaled_rrt_traj * self.scale

        return rrt_tree_map, rrt_gain_map, rrt_inc_energy_map, rrt_inc_gain_map, rrt_planned_map, rrt_traj


class RNT:
    def __init__(self, extend_area, traversability_map, conf_map, tangent_map, discorage_map):
        self.maxIter = 0
        self.sampleCount = 1
        self.extend_area = extend_area

        self.traversability_map = traversability_map
        self.conf_map = conf_map
        self.tangent_map = tangent_map
        self.discorage_map = discorage_map

    def plan(self, res, key_meters, greedy_points=None, start=None, pro_points=None, oricle_points=None, oricle_points2=None, optimize=False):
        img_h, img_w = self.traversability_map.shape

        if start is None:
            start = (int(img_h/2), int(img_w/2))
        if pro_points is None:
            pro_points = np.full((0, 2), 0)
        key_pixels = int(key_meters / res)
        canvas = np.zeros((img_h, img_w), dtype=np.uint8)
        color = 255        # Color of the circle (white)
        thickness = 1       # Thickness (-1 will fill the circle)
        cv2.circle(canvas, (start[1], start[0]), key_pixels, color, thickness)
        key_map = canvas == color
        layze = False

        sampleRadius = key_pixels / img_w
        if self.extend_area == 1:
            extend_map = self.traversability_map.copy()
        elif self.extend_area == 2:
            extend_map = np.ones_like(self.traversability_map)
        else:
            raise NotImplementedError()
        tangent_map_x = self.tangent_map[..., 0]
        tangent_map_y = self.tangent_map[..., 1]
        obj = planning.RNTWrapper(extend_map, self.conf_map, tangent_map_x, tangent_map_y, self.discorage_map, start, pro_points, np.transpose(np.nonzero(key_map)), 
                               sampleRadius, self.maxIter, self.sampleCount, layze)
        rrt_map = obj.call()

        rrt_map = rrt_map.reshape((img_h, img_w, -1)).copy()

        rrt_tree_map = rrt_map[..., 0].astype(np.int32)
        rrt_parentx_map = rrt_map[..., 1].astype(np.int32)
        rrt_parenty_map = rrt_map[..., 2].astype(np.int32)
        rrt_cost_map = rrt_map[..., 3]
        rrt_energy_map = rrt_map[..., 4]
        rrt_step_map = rrt_map[..., 5].astype(np.int32)
        rrt_inc_cost_map = rrt_map[..., 6]
        rrt_inc_energy_map = rrt_map[..., 7]
        rrt_inc_gain_map = rrt_inc_energy_map
        rrt_gain_map = 0.5 * rrt_cost_map + rrt_energy_map
        rrt_consume_map = 0.5 * rrt_cost_map - rrt_energy_map

        if not optimize:
            rrt_traj, rrt_conf = obj.trace_best()
        else:
            rrt_traj, rrt_conf = obj.trace_best_optimize()
            valid_mask = (rrt_traj[:, 0] >= 0) & (rrt_traj[:, 0] < img_h) & (rrt_traj[:, 1] >= 0) & (rrt_traj[:, 1] < img_w)
            rrt_traj = rrt_traj[valid_mask]
        rrt_traj = rrt_traj[::-1]
        print("rrt_conf", rrt_conf, rrt_conf/rrt_traj.shape[0])
        rrt_planned_map = map_traj(rrt_traj, img_h, img_w, rrt_tree_map)
        if not (oricle_points is not None or oricle_points2 is not None):
            return rrt_tree_map, rrt_gain_map, rrt_inc_energy_map, rrt_inc_gain_map, rrt_planned_map, rrt_traj
        else:
            if not optimize:
                rrt_traj2, rrt_conf2 = obj.trace_oricle(oricle_points)
            else:
                rrt_traj2, rrt_conf2 = obj.trace_oricle_optimize(oricle_points)
                valid_mask = (rrt_traj2[:, 0] >= 0) & (rrt_traj2[:, 0] < img_h) & (rrt_traj2[:, 1] >= 0) & (rrt_traj2[:, 1] < img_w)
                rrt_traj2 = rrt_traj2[valid_mask]
            rrt_traj2 = rrt_traj2[::-1]
            print("rrt_conf2", rrt_conf2, rrt_conf2/rrt_traj2.shape[0])
            rrt_planned_map2 = map_traj(rrt_traj2, img_h, img_w, rrt_tree_map)
            return (rrt_tree_map, rrt_gain_map, rrt_inc_energy_map, rrt_inc_gain_map, rrt_planned_map, rrt_traj), (rrt_tree_map, rrt_gain_map, rrt_inc_energy_map, rrt_inc_gain_map, rrt_planned_map2, rrt_traj2)


class Skeleton:
    def __init__(self, skeleton_map, traversable_map):
        self.skeleton_map = skeleton_map
        self.traversable_map = traversable_map
        
    def plan(self, start_map, cluster_frontier, greedy_points, greedy_tangents):
        dij_skeleton = Dijkstra(self.skeleton_map)
        movements_skeleton, tangents_skeleton = dij_skeleton.plan(start_map)
        smoothed_tangents_skeleton, brocasted_smoothed_tangents_skeleton, smoothed_brocasted_smoothed_tangents_skeleton = \
            smooth_and_brocast_tangent(tangents_skeleton, self.skeleton_map, self.traversable_map)
        max_accordance = -1
        max_accordance_trace_points = np.full((0, 2), 0)
        for skeleton_fid in np.unique(cluster_frontier)[1:]:
            skeleton_fpoint = np.transpose(np.nonzero(cluster_frontier == skeleton_fid))[0]
            trace_points = []
            trace_tangents = []
            current = tuple(skeleton_fpoint)
            path_points, path_tangents = dij_skeleton.back_tracing(movements_skeleton, smoothed_tangents_skeleton, current)
            if path_points.shape[0] == 0:
                break
            if greedy_points.shape[0] <= 1:
                break
            trace_points = path_points.copy()
            trace_tangents = -path_tangents

            # import matplotlib.pyplot as plt
            # fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
            # ax = axes.ravel()
            # ax[0].imshow(self.skeleton_map)
            # ax[1].imshow(start_map)
            # ax[2].imshow(cluster_frontier)
            # trace_points_map = np.full(cluster_frontier.shape, False)
            # trace_points_map[trace_points[:, 0], trace_points[:, 1]] = True
            # ax[3].imshow(trace_points_map)
            # plt.show()
            # greedy_trace_min = min(min(len(greedy_tangents), 20), min(len(trace_tangents), 20))
            # cut_g, cut_t = greedy_tangents[greedy_tangents.shape[0]-greedy_trace_min:], trace_tangents[trace_tangents.shape[0]-greedy_trace_min:]
            dist_trace_greedy = np.linalg.norm(trace_points[:, None, :] - greedy_points[None, :-1, :], axis=-1)  # greedy_points more than greedy_tangents 1
            cut_g, cut_t = greedy_tangents[dist_trace_greedy.argmin(-1)], trace_tangents.copy()
            accordance = (cut_g*cut_t).sum(axis=-1).mean(0)
            if accordance > max_accordance:
                max_accordance = accordance
                max_accordance_trace_points = trace_points.copy()
        return np.concatenate([[[self.traversable_map.shape[0]//2, self.traversable_map.shape[1]//2]], max_accordance_trace_points])


from common.coordinate import Convertor
class SkeletonScoring:
    last_pub_global = None
    QUEUE_SIZE = 5
    queue_x_, queue_y_, queue_z_ = [], [], []

    def __init__(self, img_h, img_w, res, lidar_range):
        self.img_h = img_h
        self.img_w = img_w
        self.res = res
        self.lidar_range = lidar_range

    def prepare(self, frame_id, pose, cluster_frontier, greedy_points, greedy_tangents, binary_traversable, output_dir):
        # import matplotlib.pyplot as plt
        # plt.plot(greedy_points[:, 0], greedy_points[:, 1])
        # plt.show()
        current_list = []
        for skeleton_fid in np.unique(cluster_frontier)[1:]:
            skeleton_fpoint = np.transpose(np.nonzero(cluster_frontier == skeleton_fid))[0]
            current = tuple(skeleton_fpoint)
            current_list.append(current)
        current_list = np.array(current_list)
        write_mode = "w"
        if not os.path.exists(os.path.join(output_dir, "SkeletonScoring-pose")):
            os.makedirs(os.path.join(output_dir, "SkeletonScoring-pose"))
        with open(os.path.join(output_dir, "SkeletonScoring-pose", f"{frame_id}.txt"), write_mode) as f:
            f.write(' '.join(map(str, pose.reshape(-1).tolist())) + '\n')
        if not os.path.exists(os.path.join(output_dir, "SkeletonScoring-waypoint")):
            os.makedirs(os.path.join(output_dir, "SkeletonScoring-waypoint"))
        with open(os.path.join(output_dir, "SkeletonScoring-waypoint", f"{frame_id}.txt"), write_mode) as f:
            f.write(' '.join(map(str, current_list.reshape(-1).tolist())) + '\n')
        if not os.path.exists(os.path.join(output_dir, "SkeletonScoring-osmpoint")):
            os.makedirs(os.path.join(output_dir, "SkeletonScoring-osmpoint"))
        with open(os.path.join(output_dir, "SkeletonScoring-osmpoint", f"{frame_id}.txt"), write_mode) as f:
            f.write(' '.join(map(str, greedy_points.reshape(-1).tolist())) + '\n')
        if not os.path.exists(os.path.join(output_dir, "SkeletonScoring-osmtangent")):
            os.makedirs(os.path.join(output_dir, "SkeletonScoring-osmtangent"))
        with open(os.path.join(output_dir, "SkeletonScoring-osmtangent", f"{frame_id}.txt"), write_mode) as f:
            f.write(' '.join(map(str, greedy_tangents.reshape(-1).tolist())) + '\n')
        if not os.path.exists(os.path.join(output_dir, "SkeletonScoring-traversable")):
            os.makedirs(os.path.join(output_dir, "SkeletonScoring-traversable"))
        np.save(os.path.join(output_dir, "SkeletonScoring-traversable", f"{frame_id}.npy"), binary_traversable)

    #  // 修改距离得分计算：距离自身越远，得分越高
    def calculateDistanceScore(self, pose, waypoint):
        robot_state = pose[:2, 3]
        distance = np.linalg.norm(waypoint - robot_state)
        return distance

    # // 修改OSM距离得分计算：距离OSM waypoint越近，得分越高
    def calculateOsmDistanceScore(self, waypoint, osm_waypoints):
        # // 如果osm_waypoints为空，则返回默认得分，例如0
        if osm_waypoints.shape[0] == 0:
            return 0.0

        # // 获取osm waypoints的最后一个点
        osm_waypoint = osm_waypoints[-1]

        distance = np.linalg.norm(waypoint - osm_waypoint)
        return 1.0 / (1.0 + distance);  # // 使用距离的倒数作为得分

    # // 修改角度得分计算：夹角越小，得分越高
    def calculateAngleScore(self, pose, waypoint, osm_waypoints):
        robot_state = pose[:2, 3]
        # // 如果osm_waypoints为空，则返回默认得分，例如0
        if osm_waypoints.shape[0] == 0:
            return 0.0
        
        # // 获取osm waypoints的最后一个点
        osm_waypoint = osm_waypoints[-1]

        # // 计算两个向量的点积
        dot_product = (waypoint[0] - robot_state[0]) * (osm_waypoint[0] - robot_state[0]) + (waypoint[1] - robot_state[1]) * (osm_waypoint[1] - robot_state[1])

        # // 计算两个向量的长度
        length1 = np.linalg.norm(waypoint - robot_state)
        length2 = np.linalg.norm(osm_waypoint - robot_state)

        # // 计算两个向量之间的夹角
        cos_angle = dot_product / (length1 * length2)

        angle = np.arccos(cos_angle)

        # // 使用角度的倒数作为得分
        return 1.0 / (1.0 + angle)

    def calculateChangeScore(self, waypoint, osm_waypoints, last_pub_roadwaypoint):
        if osm_waypoints.shape[0] < 2:
            return 0.0

        dx1 = waypoint[0] -last_pub_roadwaypoint[0]
        dy1 = waypoint[1] -last_pub_roadwaypoint[1]
        # // ROS_INFO("!!!!!!!!!!!!!!!!!!!!!!!!!!!!last waypoint: %f,%f",last_pub_roadwaypoint.x,last_pub_roadwaypoint.y);

        dx2 = osm_waypoints[-1][0] - osm_waypoints[-2][0]
        dy2 = osm_waypoints[-1][1] - osm_waypoints[-2][1]

        print("dx1 dy1 dx2 dy2 : %f,%f,%f,%f",dx1,dy1,dx2,dy2)

        dot_product = dx1 * dx2 + dy1 * dy2

        length1 = (dx1 * dx1 + dy1 * dy1) ** .5
        length2 = (dx2 * dx2 + dy2 * dy2) ** .5

        if length1 == 0 or length2 == 0:
            return 1.0

        cos_angle = dot_product / (length1 * length2)
        angle = np.arccos(cos_angle)

        # // Convert the angle to degrees
        # double angle_in_degrees = angle * 180.0 / M_PI;

        # // ROS_INFO("Angle in degrees: %f", angle_in_degrees);

        # // if(angle_in_degrees>70)
        # // {
        # //   return 0;
        # // }

        return 1.0 / (1.0 + angle)

    def normalize_scores(self, scores):
        if len(scores) == 0:
            return np.array([])
        # // 标准化得分
        scores = (scores - np.array(scores).min()) / (np.array(scores).max() - np.array(scores).min())
        scores = np.where(np.isnan(scores), 0, scores)
        return scores

    def publishWaypoint(self, waypoint):
        # // 将新的数据添加到队列的末尾
        self.queue_x_.append(waypoint[0])
        self.queue_y_.append(waypoint[1])

        # // 如果队列的大小超过了预设的大小，就删除队列的头部数据
        if len(self.queue_x_) > self.QUEUE_SIZE:
            self.queue_x_ = self.queue_x_[1:]
            self.queue_y_ = self.queue_y_[1:]

        # // 计算队列中所有值的平均值
        avg_x = np.array(self.queue_x_).mean()
        avg_y = np.array(self.queue_y_).mean()

        return np.array([avg_x, avg_y])

    def to_global(self, points, pose):
        if points.ndim == 1:
            single_point = True
        else:
            single_point = False
        if single_point:
            points = points[None]
        convertor = Convertor(img_h=self.img_h, img_w=self.img_w, res=self.res, lidar_range=self.lidar_range)
        points = convertor.matrix2car(points)
        R_90 = np.array([
            [0,  1],
            [-1, 0]
        ])
        points = points @ R_90.T  # Apply the rotation matrix
        hom_points_i = np.zeros((points.shape[0], 4))
        hom_points_i[:, :2] = points
        hom_points_i[:, 3] = 1
        points = pose.dot(hom_points_i.T).T
        points = points[:, :2]
        if single_point:
            points = points[0]
        return points

    def to_local(self, points, pose):
        if points.ndim == 1:
            single_point = True
        else:
            single_point = False
        if single_point:
            points = points[None]
        hom_points_i = np.zeros((points.shape[0], 4))
        hom_points_i[:, :2] = points
        hom_points_i[:, 3] = 1
        points = np.linalg.inv(pose).dot(hom_points_i.T).T
        points = points[:, :2]
        R_90_counter = np.array([
            [0, -1],
            [1,  0]
        ])
        points = points @ R_90_counter.T
        convertor = Convertor(img_h=self.img_h, img_w=self.img_w, res=self.res, lidar_range=self.lidar_range)
        points = convertor.car2matrix(points)
        if single_point:
            points = points[0]
        return points

    def matrix_list2map(self, points):
        convertor = Convertor(img_h=self.img_h, img_w=self.img_w, res=self.res, lidar_range=self.lidar_range)
        points = convertor.matrix_list2map(points)
        return points
    
    def bezier_curve(self, p0, p1, p2, num_points=100):
        """
        Generate points for a quadratic Bézier curve.
        
        Parameters:
            p0 (tuple): Start point (x, y).
            p1 (tuple): Control point (x, y).
            p2 (tuple): End point (x, y).
            num_points (int): Number of points to compute along the curve.
        
        Returns:
            numpy.ndarray: Array of shape (num_points, 2) containing the points on the curve.
        """
        t = np.linspace(0, 1, num_points)[:, None]
        curve = (1 - t)**2 * np.array(p0)[None] + 2 * (1 - t) * t * np.array(p1)[None] + t**2 * np.array(p2)[None]
        return curve

    def connect_with_bezier_curves(self, p0, p2, num_curves=5, control_variation=0.5):
        """
        Connect two points with multiple Bézier curves by varying control points.
        
        Parameters:
            p0 (tuple): Start point (x, y).
            p2 (tuple): End point (x, y).
            num_curves (int): Number of Bézier curves to generate.
            control_variation (float): Maximum variation of control points around the midpoint.
        
        Returns:
            list: A list of numpy arrays, each containing points for a Bézier curve.
        """
        curves = []
        midpoint = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2)

        for i in range(num_curves):
            # Randomly vary the control point around the midpoint
            control_x = midpoint[0] + np.random.uniform(-control_variation, control_variation)
            control_y = midpoint[1] + np.random.uniform(-control_variation, control_variation)
            p1 = (control_x, control_y)
            
            # Generate the Bézier curve
            curve = self.bezier_curve(p0, p1, p2)
            curves.append(curve)
        
        return curves

    def plan(self, frame_id, output_dir):
        idx = int(frame_id)
        idxs = np.arange(idx - 20, idx+1)
        # frame_id_i = str(idx).zfill(len(frame_id))
        frame_ids = [item[:-4] for item in os.listdir(os.path.join(output_dir, "SkeletonScoring-pose"))]
        frame_ids.sort()
        valid_idx = idxs[(idxs >= 0) & (idxs < len(frame_ids))]
        valid_frames = np.array(frame_ids)[valid_idx]
        with open(os.path.join(output_dir, "SkeletonScoring-pose", f"{frame_id}.txt"), "r") as f:
            pose = np.fromstring(f.readline().strip(), dtype=float, sep=' ').reshape(4, 4)
        poses = []
        for item in valid_frames:
            with open(os.path.join(output_dir, "SkeletonScoring-pose", f"{item}.txt"), "r") as f:
                poses.append(np.fromstring(f.readline().strip(), dtype=float, sep=' ').reshape(4, 4))
        waypoints = []
        for item in valid_frames:
            with open(os.path.join(output_dir, "SkeletonScoring-waypoint", f"{item}.txt"), "r") as f:
                waypoints.append(np.fromstring(f.readline().strip(), dtype=float, sep=' ').reshape(-1, 2))
        with open(os.path.join(output_dir, "SkeletonScoring-osmpoint", f"{frame_id}.txt"), "r") as f:
            osmpoint = np.fromstring(f.readline().strip(), dtype=float, sep=' ').reshape(-1, 2)
        binary_traversable = np.load(os.path.join(output_dir, "SkeletonScoring-traversable", f"{frame_id}.npy"))
        global_waypoints = []
        for wp, rp in zip(waypoints, poses):
            global_waypoints.append(self.to_global(wp, rp))
        global_waypoints = np.concatenate(global_waypoints, axis=0)
        global_osmpoint = self.to_global(osmpoint, pose)

        print("pose, global_waypoints, global_osmpoint", pose, global_waypoints, global_osmpoint)

        dis_OSM_scores = []
        distance_scores = []
        angle_scores = []
        change_scores = []
        for wp in global_waypoints:
            dis_OSM_score = self.calculateOsmDistanceScore(wp, global_osmpoint)
            distance_score = self.calculateDistanceScore(pose, wp)
            angle_score = self.calculateAngleScore(pose, wp, global_osmpoint)
            osmpoint_mv = global_osmpoint.copy()
            if self.last_pub_global is not None:
                change_score = self.calculateChangeScore(wp, osmpoint_mv, self.last_pub_global)
            else:
                change_score = 0

            print("Change Score: %f", change_score)

            dis_OSM_scores.append(dis_OSM_score)
            distance_scores.append(distance_score)
            angle_scores.append(angle_score)
            change_scores.append(change_score)
        
        # print("dis_OSM_scores, distance_scores, angle_scores, change_scores", dis_OSM_scores, distance_scores, angle_scores, change_scores)

        dis_OSM_scores = self.normalize_scores(dis_OSM_scores)
        distance_scores = self.normalize_scores(distance_scores)
        angle_scores = self.normalize_scores(angle_scores)
        change_scores = self.normalize_scores(change_scores)

        # print("normalized dis_OSM_scores, distance_scores, angle_scores, change_scores", dis_OSM_scores, distance_scores, angle_scores, change_scores)

        K_dis_OSM_scores = 1.2
        K_distance_scores = 0.7
        K_angle_scores = 1.4
        K_change_scores = 1.0

        # // 计算总得分并存储
        waypoint_scores = []
        for i in range(global_waypoints.shape[0]):
            total_score = K_dis_OSM_scores * dis_OSM_scores[i] + K_distance_scores * distance_scores[i] + K_angle_scores * angle_scores[i] + K_change_scores * change_scores[i]
            waypoint_scores.append(total_score)
        waypoint_scores = np.array(waypoint_scores)

        # print("global_waypoints, waypoint_scores", global_waypoints, waypoint_scores)

        argsort = waypoint_scores.argsort()[::-1]
        global_waypoints, waypoint_scores = global_waypoints[argsort], waypoint_scores[argsort]
        dis_OSM_scores = dis_OSM_scores[argsort]
        distance_scores = distance_scores[argsort]
        angle_scores = angle_scores[argsort]
        change_scores = change_scores[argsort]

        # print("sorted global_waypoints, waypoint_scores", global_waypoints, waypoint_scores)
        # print("sorted dis_OSM_scores, distance_scores, angle_scores, change_scores", dis_OSM_scores, distance_scores, angle_scores, change_scores)

        print("calculateWaypointScores2")

        latest_osm_waypoint = global_osmpoint[-1]

        if len(global_waypoints) == 0:
            print("Score waypoint is empty, publishing OSM waypoint instead.")
            # self.publishWaypoint(latest_osm_waypoint)
            final_point = latest_osm_waypoint
        else:
            # // 提取前1个得分最高的waypoints
            transformed_waypoint_top, waypoint_score_top = global_waypoints[0], waypoint_scores[0]
            # visualize_waypoint_scores(top_scores_waypoints)

            # print("transformed_waypoint_top, waypoint_score_top", transformed_waypoint_top, waypoint_score_top)

            dist = np.linalg.norm(latest_osm_waypoint - transformed_waypoint_top)
            osm_score_MAX_DISTANCE = 40
            if dist > osm_score_MAX_DISTANCE:
                print("Distance between OSM waypoint and Score waypoint too large, publishing OSM waypoint instead.")
                # self.publishWaypoint(latest_osm_waypoint)
                final_point = latest_osm_waypoint
            else:
                midpoint = transformed_waypoint_top.copy()
                self.last_pub_global = midpoint.copy()
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!publish road waypoint: %f,%f",midpoint[0],midpoint[1])
                # self.publishWaypoint(midpoint)
                final_point = midpoint

        robot_state = pose[:2, 3]
        local_global_osmpoint = self.to_local(global_osmpoint, pose)
        local_global_waypoints = self.to_local(global_waypoints, pose)
        local_final_point = self.to_local(final_point, pose)
        local_robot_state = self.to_local(robot_state, pose)
        local_curves = self.connect_with_bezier_curves(local_robot_state, local_final_point, num_curves=100, control_variation=100)
        vis_local_global_osmpoint = np.stack([local_global_osmpoint[:, 1], local_global_osmpoint[:, 0]], axis=-1)
        vis_local_global_waypoints = np.stack([local_global_waypoints[:, 1], local_global_waypoints[:, 0]], axis=-1)
        vis_local_final_point = np.stack([local_final_point[1], local_final_point[0]], axis=-1)
        vis_local_robot_state = np.stack([local_robot_state[1], local_robot_state[0]], axis=-1)
        vis_local_curves = [np.stack([curve[:, 1], curve[:, 0]], axis=-1) for curve in local_curves]
        # local_global_osmpoint_map = self.matrix_list2map(local_global_osmpoint)
        # local_global_waypoints_map = self.matrix_list2map(local_global_waypoints)
        # local_final_point_map = self.matrix_list2map(local_final_point[None])[0]
        # local_robot_state_map = self.matrix_list2map(local_robot_state[None])[0]
        
        curve_score = []
        for curve in local_curves:
            # Calculate the distances between consecutive points
            curve_distance = np.sum(np.linalg.norm(np.diff(curve, axis=0), axis=1))
            curve_grid = curve.astype(int)
            curve_grid = curve_grid[((curve_grid >= 0) & (curve_grid < min(self.img_h, self.img_w))).all(1)]
            curve_traversable = np.sum(binary_traversable[curve_grid.T[0], curve_grid.T[1]])
            curve_score.append(1 / curve_distance + curve_traversable / curve_grid.shape[0])
        curve_score = np.array(curve_score)
        local_curves_best = local_curves[curve_score.argmax()]
        vis_local_curves_best = vis_local_curves[curve_score.argmax()]

        import matplotlib.pyplot as plt
        from common.visulization import Visulizater
        axes = plt.gca()
        axes.imshow(binary_traversable)
        axes.scatter(vis_local_global_osmpoint[:, 0], vis_local_global_osmpoint[:, 1], s=10., color="red")
        axes.scatter(vis_local_global_waypoints[:, 0], vis_local_global_waypoints[:, 1], s=10., color="green")
        for i in range(waypoint_scores.shape[0]):
            # print("i dis_OSM_scores, distance_scores, angle_scores, change_scores", i, dis_OSM_scores[i], distance_scores[i], angle_scores[i], change_scores[i])
            pos = (vis_local_global_waypoints[i, 0]+np.random.rand(), vis_local_global_waypoints[i, 1]+np.random.rand())
            # axes.annotate(f"{waypoint_scores[i]:.2f}\n{K_dis_OSM_scores * dis_OSM_scores[i]:.2f}\n{K_distance_scores * distance_scores[i]:.2f}\n{K_angle_scores * angle_scores[i]:.2f}\n{K_change_scores * change_scores[i]:.2f}\n", pos, fontsize=10)
            axes.annotate(f"{waypoint_scores[i]:.2f}\n", pos, fontsize=10)
        axes.scatter(vis_local_final_point[0], vis_local_final_point[1], s=40., color="blue")
        axes.scatter(vis_local_robot_state[0], vis_local_robot_state[1], s=40., color="red")
        plt.gca().set_aspect('equal')
        for curve in vis_local_curves:
            plt.plot(curve[:, 0], curve[:, 1], label="Bezier Curve")
        plt.plot(vis_local_curves_best[:, 0], vis_local_curves_best[:, 1], linewidth=4.0)
        # visulizater = Visulizater(np.full((1, 3), 0.))
        # axes.imshow(visulizater.stack_layer(
        #     visulizater.binary_label_vis(binary_traversable), 
        #     (local_global_osmpoint_map,local_global_waypoints_map,local_final_point_map,local_robot_state_map), 
        #     fg_color_list=(np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.]), np.array([1.,0.,0.]))))
        # plt.show()
        if not os.path.exists(os.path.join(output_dir, "SkeletonScoring-vis")):
            os.makedirs(os.path.join(output_dir, "SkeletonScoring-vis"))
        plt.savefig(os.path.join(output_dir, "SkeletonScoring-vis", f"{frame_id}.png"))
        plt.clf()

        if int(frame_id) != 0:
            write_mode = "a"
        else:
            write_mode = "w"
        with open(os.path.join(output_dir, "SkeletonScoring-result.txt"), write_mode) as f:
            f.write(' '.join(map(str, local_curves_best.reshape(-1).tolist())) + '\n')

        return local_final_point, local_curves_best

    def readfile(self, frame_id, output_dir):
        idx = int(frame_id)
        with open(os.path.join(output_dir, "SkeletonScoring-result.txt"), "r") as f:
            lines = f.readlines()
            path = np.fromstring(lines[idx].strip(), dtype=float, sep=' ').reshape(-1, 2)
            return path.astype(int)


class Straight:
    def __init__(self, traversable_map):
        self.traversable_map = traversable_map
        
    def plan(self, greedy_points):
        straight = np.full((0, 2), 0)
        if greedy_points.shape[0] > 0:
            middle_greedy_point = greedy_points[cdist([greedy_points.mean(axis=0)], greedy_points).argmin()]
            B = tuple(middle_greedy_point)
            A = (self.traversable_map.shape[0] // 2, self.traversable_map.shape[1] // 2)
            rr, cc = line(A[0], A[1], B[0], B[1])  # Note: skimage.draw.line uses (row, col) order
            straight = np.array(list(zip(rr, cc)))
        return straight


import math


def euclidean_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def find_intersection(p1, p2, center, radius):
    """
    Find the intersection of the line segment p1-p2 with the circle centered at 'center' with 'radius'.
    p1 is inside the circle, p2 is outside.
    """
    # Vector from p1 to p2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    
    # Vector from center to p1
    fx, fy = p1[0] - center[0], p1[1] - center[1]
    
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radius * radius
    
    # Find discriminant
    discriminant = b * b - 4 * a * c
    if discriminant < 0:
        print("discriminant", discriminant)
        return None  # No intersection
    
    discriminant = math.sqrt(discriminant)
    
    # Find two solutions t1 and t2
    t1 = (-b - discriminant) / (2 * a)
    t2 = (-b + discriminant) / (2 * a)
    
    # Return the point on the segment (closest to p1, but within the segment)
    epsilon = 1e-6
    if 0-epsilon <= t1 <= 1+epsilon:
        return (p1[0] + t1 * dx, p1[1] + t1 * dy)
    elif 0-epsilon <= t2 <= 1+epsilon:
        return (p1[0] + t2 * dx, p1[1] + t2 * dy)
    else:
        print("t1, t2", t1, t2)
        return None


def trim_path(traj, radius, zero_center=False):
    if traj.shape[0] == 0:
        return traj.copy()
    else:
        if not zero_center:
            center = traj[0]
        else:
            center = np.zeros_like(traj[0])
        new_traj = [traj[0]]
        for i in range(1, traj.shape[0]):
            parent = traj[i-1]
            node = traj[i]
            if euclidean_distance(node, center) <= radius:
                new_traj.append(node)
            else:
                intersection = find_intersection(parent, node, center, radius)
                assert intersection is not None, "parent, node, center, radius: {} {} {} {}".format(parent, node, center, radius)
                new_traj.append(list(intersection))
                break
        return np.stack(new_traj, axis=0)


class PathHandler:
    def __init__(self, img_h, img_w, res, meters) -> None:
        self.img_h = img_h
        self.img_w = img_w
        self.res = res
        self.meters = meters

    def handle_limit(self, traj_points):
        return trim_path(traj_points, self.meters // self.res)

    def handle_empty(self, traj_points):
        return traj_points.copy() if traj_points.shape[0] > 0 else np.array([[self.img_h // 2, self.img_w // 2]])
    
    def handle(self, traj_points):
        return self.handle_empty(self.handle_limit(traj_points))
    

if __name__ == "__main__":
    from common.polar import polar_to_cartesian
    from common.visulization import Visulizater
    from common.coordinate import Convertor
    from matplotlib import pyplot as plt
    shape = (400, 400)
    traversability_map = np.full(shape, 0)
    conf_map = np.full(shape, 1.)
    # tangent_map = np.full(shape+(2,), 1.)
    polar = np.full(shape, np.pi)
    polar[:shape[0]//8] -= np.pi/8
    tangent_map = np.stack(list(polar_to_cartesian(np.full(shape, 1.), polar)), axis=-1)
    discorage_map = np.full(shape, 0.)
    rrt_planner = RRT(2, 2, traversability_map, conf_map, tangent_map, discorage_map)
    rnt_planner = RNT(2, traversability_map, conf_map, tangent_map, discorage_map)
    rrt_tree_map, rrt_gain_map, rrt_inc_energy_map, rrt_inc_gain_map, rrt_planned_map, rrt_traj = rrt_planner.plan(1, shape[0]//2-shape[0]//16)
    rnt_tree_map, rnt_gain_map, rnt_inc_energy_map, rnt_inc_gain_map, rnt_planned_map, rnt_traj = rnt_planner.plan(1, shape[0]//2-shape[0]//16, optimize=False)
    rnt_tree_map2, rnt_gain_map2, rnt_inc_energy_map2, rnt_inc_gain_map2, rnt_planned_map2, rnt_traj2 = rnt_planner.plan(1, shape[0]//2-shape[0]//16, optimize=True)
    visulizater = Visulizater(np.full((1, 3), 0.))
    convertor = Convertor(img_h=shape[0], img_w=shape[1])
    figure = plt.figure(figsize=(20, 10))
    color_tree = np.array([0/255,128/255,255/255])
    colored_bg = visulizater.tangent_vis(-tangent_map, pltcm='hsv')[..., :3]
    colored_bg = np.ones_like(colored_bg)
    axesimg = figure.add_subplot(1, 3, 1)
    axesimg.axis("off")
    axesimg.imshow(visulizater.stack_layer(
        colored_bg, 
        (rrt_tree_map > 0,)+convertor.matrix_list2map(rrt_traj, edge=True)[::-1], 
        fg_color_list=(color_tree,)+(np.array([1.,0.,0.]), np.array([1.,0.,0.]))[::-1]))
    axesimg = figure.add_subplot(1, 3, 2)
    axesimg.axis("off")
    axesimg.imshow(visulizater.stack_layer(
        colored_bg, 
        (rnt_tree_map > 0,)+convertor.matrix_list2map(rnt_traj, edge=True)[::-1], 
        fg_color_list=(color_tree,)+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
    axesimg = figure.add_subplot(1, 3, 3)
    axesimg.axis("off")
    axesimg.imshow(visulizater.stack_layer(
        colored_bg, 
        (rnt_tree_map2 > 0,)+convertor.matrix_list2map(rnt_traj2, edge=True)[::-1], 
        fg_color_list=(color_tree,)+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
    plt.show()