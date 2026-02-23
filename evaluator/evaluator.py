import os
from multiprocessing import Pool

import yaml

import numpy as np
from scipy.spatial.distance import euclidean, cdist
from scipy.ndimage import uniform_filter
from scipy.interpolate import splprep, splev
from fastdtw import fastdtw
import torch
import matplotlib as mpl
from matplotlib import pyplot as plt
from torchvision.utils import save_image
from torchvision.transforms import ToTensor
import cv2

import itertools
import logging
import numpy as np
import os
from collections import OrderedDict
from PIL import Image

from detectron2.utils.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import (
    DatasetEvaluator, SemSegEvaluator, COCOPanopticEvaluator
)
from detectron2.utils.comm import all_gather, is_main_process, synchronize

from common.timer import Timer
from common.polar import *
from common.visulization import Visulizater
from common.peroidic import peroidic_np
from common import planning
from common.traversibility import nearest_cluster, estimate_frontier_and_skeleton
from common.coordinate import Convertor
from common.envs import *
from common.utils import trans_by_switch
from common.edt_tangent import edt_with_tangent
from deploy.end2end import End2End
from .metrics import interp_l2_distance_3, hausdorff_dist
from .plotting import plot_waypoint
from .plotting2 import plot_histogram

from lidardet.ops.lidar_bev import bev


logger = logging.getLogger(__name__)


def get_pred(tangent_init_polar, pred):
    return tangent_init_polar + pred
    # return np.zeros_like(tangent_init_polar) + pred


def save_tensor_as_image(tensor, filename):
    """
    Save a NumPy tensor with shape H x W x 3 as a color image.

    Parameters:
        tensor (numpy.ndarray): The input tensor with shape H x W x 3.
                                Values should be in the range [0, 255] for uint8 or [0, 1] for float.
        filename (str): The output filename, including the extension (e.g., 'image.png').

    Returns:
        None
    """
    # Ensure the tensor has the correct shape
    if tensor.ndim != 3 or tensor.shape[2] != 3:
        raise ValueError("Input tensor must have shape H x W x 3.")
    
    # Check the data type and range
    if tensor.dtype == np.float32 or tensor.dtype == np.float64:
        # If the array is in the range [0, 1], scale to [0, 255]
        if tensor.max() <= 1.0 and tensor.min() >= 0.0:
            tensor = (tensor * 255).astype(np.uint8)
        else:
            raise ValueError("Float tensors must have values in the range [0, 1].")
    elif tensor.dtype != np.uint8:
        raise ValueError("Tensor must have dtype uint8 or float in range [0, 1].")

    # Convert the NumPy array to a PIL Image and save it
    image = Image.fromarray(tensor)
    image.save(filename)


eval_meters = [10, 20]
# eval_meters = [5, 10]
class TrajectoryEvaluator(SemSegEvaluator):
    def __init__(self, dataset_name, distributed, output_dir):
        super().__init__(
            dataset_name,
            distributed=distributed,
            output_dir=output_dir,
        )
        self.convertor = Convertor()
        color_map = MetadataCatalog.get(dataset_name).get("color_map")
        self.colormap = np.zeros_like(color_map)
        self.colormap[..., 0] = color_map[..., 2]
        self.colormap[..., 1] = color_map[..., 1]
        self.colormap[..., 2] = color_map[..., 0]
        self.visulizater = Visulizater(self.colormap)
        self.end2end = None

    def reset(self):
        super(TrajectoryEvaluator, self).reset()
        self._metas = []
        self._commons = []
        self._metrics = []
        self._waypoints = []
        self._trajs = []
        # # pooling
        # self.pool = Pool(processes=4)  # Create a pool of 4 worker processes

    def rrt_planner(self, extend_mode, extend_area, label_traversable, tangent_map, conf_map=None, encorage_map=None, tangent_map2=None):
        if conf_map is None:
            conf_map = np.full(label_traversable.shape, 1.)
        if encorage_map is None:
            discorage_map = np.full(label_traversable.shape, 0.)
        else:
            discorage_map = 1 - encorage_map
        if tangent_map2 is None:
            tangent_map2 = np.zeros_like(tangent_map)
        return planning.ScaleRRT(extend_mode, extend_area, label_traversable != 0, conf_map, -tangent_map, -tangent_map2, discorage_map)

    def rnt_planner(self, extend_area, label_traversable, tangent_map, conf_map=None, encorage_map=None):
        if conf_map is None:
            conf_map = np.full(label_traversable.shape, 1.)
        if encorage_map is None:
            discorage_map = np.full(label_traversable.shape, 0.)
        else:
            discorage_map = 1 - encorage_map
        return planning.RNT(extend_area, label_traversable != 0, conf_map, -tangent_map, discorage_map)

    def ngm_optimizer(self, label_traversable, tangent_map):
        return planning.NGMOptimizer(label_traversable != 0, -tangent_map)

    def key_traj(self, traj, key_meters, mode):
        if traj.shape[0] == 0:
            key_traj = traj.copy()
        else:
            if mode == 'car':
                key_mask = np.linalg.norm(traj - traj[0], axis=-1) <= key_meters
            elif mode == 'matrix':
                car_traj = self.convertor.matrix2car(traj)
                key_mask = np.linalg.norm(car_traj - car_traj[0], axis=-1) <= key_meters
            traj_idx = np.arange(traj.shape[0])
            key_traj_idx = traj_idx[key_mask]
            key_traj = traj[key_traj_idx[key_traj_idx == np.arange(key_traj_idx.shape[0])]]
        return key_traj

    def traj_diff(self, target_traj, target_points, pred_points, key_meters):
        pred = self.convertor.matrix2car(pred_points)
        target_traj = self.key_traj(target_traj, key_meters, 'car')
        target = target_traj.copy() if target_traj.shape[0] > 0 else pred.copy()

        distance, path = fastdtw(pred, target, dist=euclidean)
        return {f"diff_{key_meters}m": distance}

    def traj_dict(self, target_traj, target_points, pred_points, key_meters):  # L 2, L 2
        pred = self.convertor.matrix2car(pred_points)
        target_traj = self.key_traj(target_traj, key_meters, 'car')
        target = target_traj.copy() if target_traj.shape[0] > 0 else pred.copy()

        pred_interp, target_interp, metric = interp_l2_distance_3(pred, target, np.linalg.norm(target[-1]))  # target cut
        ADE = metric['ADE']
        FDE = metric['FDE']
        HitRate = 1 if metric['MAX'] < 2 else 0
        HD = hausdorff_dist(pred, target)
        return {
            f'ADE_{key_meters}m': ADE,
            f'FDE_{key_meters}m': FDE,
            f'HitRate_{key_meters}m': HitRate,
            f'HD_{key_meters}m': HD,
        }, pred_interp, target_interp

    def traj_coverage(self, target_traj, target_points, pred_points, key_meters):
        pred = self.convertor.matrix2car(pred_points)
        target_traj = self.key_traj(target_traj, key_meters, 'car')
        target = target_traj.copy() if target_traj.shape[0] > 0 else pred.copy()
        
        cover_meters = 2
        pred_target = np.linalg.norm(pred[:, None, :] - target[None, :, :], axis=-1)  # pred by target
        pred_cloest = pred_target.min(axis=-1)
        target_cloest = pred_target.min(axis=0)
        pred_cover = pred_cloest <= cover_meters
        target_cover = target_cloest <= cover_meters
        # TP = pred_cover.sum()
        # FP = (~pred_cover).sum()
        # FN = (~target_cover).sum()
        precision = pred_cover.sum()/pred_cover.shape[0]
        recall = target_cover.sum()/target_cover.shape[0]
        F1 = 2*precision*recall/(precision+recall) if precision+recall != 0 else 0
        return {
            f"cover_pre_{key_meters}m": precision,
            f"cover_rec_{key_meters}m": recall,
            f"cover_F1_{key_meters}m": F1,
        }

    def traj_coverage_interp(self, pred_interp, target_interp, key_meters):
        cover_meters = 2
        ds = np.linalg.norm(target_interp-pred_interp, axis=1)
        n_target = target_interp.shape[0]
        n_pred = pred_interp.shape[0]
        n_intersection = (ds <= cover_meters).sum()
        n_union = n_target + n_pred - n_intersection
        return {
            f"cover_interp_rec_{key_meters}m": n_intersection / n_target,
            f"cover_interp_IoU_{key_meters}m": n_intersection / n_union,
        }

    def waypoint_diff(self, target_traj, target_points, pred_points, key_meters):
        pred = self.convertor.matrix2car(pred_points)
        target_traj = self.key_traj(target_traj, key_meters, 'car')
        target = target_traj.copy() if target_traj.shape[0] > 0 else pred.copy()

        pred2 = pred_points.copy()
        target_traj2 = self.key_traj(target_points, key_meters, 'matrix')
        target2 = target_traj2.copy() if target_traj2.shape[0] > 0 else pred2.copy()

        return {f"waypoint_{key_meters}m": np.linalg.norm(target[-1]-pred[-1])}, pred[-1], target[-1], pred2[-1], target2[-1]

    def cut_by_length(self, traj, length):
        cut_idx = len(traj) - 1
        for i, d in enumerate(np.cumsum(np.linalg.norm(np.diff(traj, axis=0), axis=1))):
            if d >= length:
                cut_idx = i
                break
        return traj[:cut_idx+1]

    def set_limits(self, ax):
        ### set the range of x and y
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        plot_radius = max([abs(lim - mean_)
            for lims, mean_ in ((xlim, xmean),
                (ylim, ymean))
            for lim in lims])
        ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    def get_save_dir(self, busi, key, seq_id, frame_id, format):
        pred_path = os.path.join(self._output_dir, "save", "val", busi, "sequences", seq_id, key)
        if not os.path.isdir(pred_path):
            os.makedirs(pred_path)
        return os.path.join(pred_path, "{}.{}".format(frame_id, format))

    def choose_by_centriod(self, points):
        return points[cdist([points.mean(axis=0)], points).argmin()]

    def merge_dict(self, dicts):
        new_dict = {}
        for i in range(len(dicts)):
            knv = dicts[i]
            for k, nv, in knv.items():
                for n, v in nv.items():
                    if k not in new_dict.keys():
                        new_dict[k] = {}
                    new_dict[k][n] = v
        return new_dict

    def collect_result(self, result):
        batch_metas, batch_commons, batch_metrics, batch_waypoints, batch_trajs = result
        self._metas.extend(batch_metas)
        self._commons.extend(batch_commons)
        self._metrics.extend(batch_metrics)
        self._waypoints.extend(batch_waypoints)
        self._trajs.extend(batch_trajs)

    def collect_error(self, result):
        batch_metas, batch_commons, batch_metrics, batch_waypoints, batch_trajs = result
        self._metas.extend(batch_metas)
        self._commons.extend(batch_commons)
        self._metrics.extend(batch_metrics)
        self._waypoints.extend(batch_waypoints)
        self._trajs.extend(batch_trajs)

    # def process(self, inputs, outputs):
    #     self.pool.apply_async(self.process_one_batch, args=(inputs, outputs), callback=self.collect_result, error_callback=self.collect_error)

    def process(self, inputs, outputs):
        self.collect_result(self.process_one_batch(inputs, outputs))

    def process_one_batch(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """
        batch_metas, batch_commons, batch_metrics, batch_waypoints, batch_trajs = [], [], [], [], []
        for input, output in zip(inputs, outputs):
            # key_meters = self.res * (label_mask.shape[1] / 2)
            key_meters = eval_meters[-1]

            self.res = input["res"]
            self.lidar_range = input["lidar_range"]
            self.convertor.reinit(img_h=input["training_mask"].shape[0], img_w=input["training_mask"].shape[1], res=self.res, lidar_range=self.lidar_range)

            seq_id, frame_id = input["seq_id"], input["frame_id"]
            pose = input["pose"]

            is_baseline = False
            is_ablation = True
            is_free_space_available = True

            if is_baseline:
                data_twin_root = "semantic-kitti"
                # data_twin_root = "um-osmplanner-4.livox"
                self.end2end_root = os.path.join( f'../../trajectory-prediction/cache/transformer/100/output.' + data_twin_root)
                if self.end2end is None:
                    with open(os.path.join(os.path.dirname(self.end2end_root), "../../../config/trajectory_prediction/transformer.yaml"), "r") as file:
                        self.end2end_cfg = yaml.safe_load(file)  # Use safe_load to avoid arbitrary code execution
                        res = self.end2end_cfg['voxel_size'][0] # 0.16 m / pixel
                        xmin, ymin, zmin, xmax, ymax, zmax = self.end2end_cfg['point_cloud_range']
                        lidar_range = {'Left': xmin, 'Right': xmax, 'Front': ymax, 'Back': ymin, 'Bottom': zmin, 'Top': zmax}
                        map_h = lidar_range['Front'] - lidar_range['Back']
                        map_w = lidar_range['Right'] - lidar_range['Left']
                        self.end2end = End2End(os.path.join(self.end2end_root, "model.onnx"), map_h, map_w, res, lidar_range['Left'], lidar_range['Front'])
                        self.end2end.img_h = int(map_h / res)
                        self.end2end.img_w = int(map_w / res)
                        self.end2end.lidar_range = lidar_range
                        self.end2end.lidar_range_np = np.array([xmin, xmax, ymin, ymax, zmin, zmax])
                        self.end2end.res = res
                        self.end2end.sensor_height = self.end2end_cfg['sensor_height']
                        self.end2end.road_width = self.end2end_cfg['data']['val']['road_width']

            bev_map = input["bev_map"].numpy()
            training_mask = input["training_mask"].numpy()
            bev_label = input["bev_label"].numpy()
            label_mask = input["label_mask"].numpy()
            label_padded = input["label_padded"].numpy()
            binary_traversable = input["binary_traversable"].numpy()
            full_bev_label = input["full_bev_label"].numpy()
            full_label_mask = input["full_label_mask"].numpy()
            full_label_padded = input["full_label_padded"].numpy()
            full_binary_traversable = input["full_binary_traversable"].numpy()

            tangent_init = input["tangent_init"].numpy()
            tangent_init_polar = cartesian_to_polar(tangent_init[..., 0], tangent_init[..., 1])[1] / np.pi
            distance_init = input["distance_init"].numpy()

            output_regression = output["regression"][0].to(self._cpu_device)
            pred = peroidic_np(np.array(output_regression.detach(), dtype=np.float32), norm=False, abs=False)

            tangent_map_pred_polar = peroidic_np(get_pred(tangent_init_polar, pred), norm=False, abs=False)
            tangent_map_pred_x, tangent_map_pred_y = polar_to_cartesian(label_mask, tangent_map_pred_polar * np.pi)
            tangent_map_pred = np.stack([tangent_map_pred_x, tangent_map_pred_y], axis=-1)

            timer = Timer(True)

            timer.put("tangent ours")
            tangent_map_pred_field = np.stack(list(polar_to_cartesian(np.ones_like(label_mask), tangent_map_pred_polar * np.pi)), axis=-1)
            tangent_map_pred_alpha = tangent_map_pred_field.copy()
            tangent_map_pred_beta = tangent_map_pred_field.copy()
            tangent_map_pred_beta_rnt = tangent_map_pred_field.copy()

            tangent_map_pred_beta_unifroms15 = np.stack([uniform_filter(tangent_map_pred_beta[..., 0], size=15, mode="nearest"), uniform_filter(tangent_map_pred_beta[..., 1], size=15, mode="nearest")], axis=-1)
            tangent_map_pred_beta_rnt_unifroms15 = np.stack([uniform_filter(tangent_map_pred_beta_rnt[..., 0], size=15, mode="nearest"), uniform_filter(tangent_map_pred_beta_rnt[..., 1], size=15, mode="nearest")], axis=-1)
            timer.get()

            tangent_map = input["tangent_map"].numpy()
            tangent_map_polar = cartesian_to_polar(tangent_map[..., 0], tangent_map[..., 1])[1] / np.pi

            origin = peroidic_np(tangent_map_polar - tangent_init_polar, norm=False, abs=False)
            now = peroidic_np(tangent_map_polar - tangent_map_pred_polar, norm=False, abs=False)

            eline_points = input["eline_points"].numpy()
            eline_points_map = input["eline_points_map"].numpy()
            eline_tangents = input["eline_tangents"].numpy()
            eline_tangents_map = input["eline_tangents_map"].numpy()
            curve_points = input["curve_points"].numpy()
            curve_points_map = input["curve_points_map"].numpy()
            curve_tangents = input["curve_tangents"].numpy()
            curve_tangents_map = input["curve_tangents_map"].numpy()

            greedy_point = input["greedy_point"].numpy()
            greedy_points = input["greedy_points"].numpy()
            greedy_tangents = input["greedy_tangents"].numpy()

            target_points = input["target_points"].numpy()
            target_points_map = input["target_points_map"].numpy()
            target_traj = input["target_traj"].numpy()

            timer.put("estimate_frontier_and_skeleton")
            if is_ablation:
                binary_frontier, cluster_frontier, label_pollute, baised_skeleton, baised_frontier_skeleton, skeleton_binary_traversable, skeleton_cluster_frontier = \
                    estimate_frontier_and_skeleton(label_padded, binary_traversable)
                greedy_frontier = nearest_cluster(cluster_frontier, greedy_point)
                traversablepoints = np.transpose(np.nonzero(binary_traversable))
                greedy_traversable_point = traversablepoints[cdist([greedy_point], traversablepoints).argmin()]
                greedy_traversable = self.convertor.matrix_list2map(greedy_traversable_point[None])
                full_traversablepoints = np.transpose(np.nonzero(full_binary_traversable))
                full_greedy_traversable_point = full_traversablepoints[cdist([greedy_point], full_traversablepoints).argmin()]
                full_greedy_traversable = self.convertor.matrix_list2map(full_greedy_traversable_point[None])
                skeletonpoints = np.transpose(np.nonzero(skeleton_binary_traversable))
                if len(skeletonpoints) > 0:
                    skeletonpoint = skeletonpoints[cdist([(skeleton_binary_traversable.shape[0]//2, skeleton_binary_traversable.shape[1]//2)], skeletonpoints).argmin()]
                    skeletonpoint_map = self.convertor.matrix_list2map(skeletonpoint[None])
                else:
                    skeletonpoint_map = np.full(skeleton_binary_traversable.shape, False)
            timer.get()

            timer.flush()

            timer.put("tangent baselines")
            if is_ablation:
                if is_free_space_available:
                    tangent_map_ogm = np.zeros_like(eline_tangents_map)
                tangent_map_simple = eline_tangents_map.copy()
                tangent_map_naive = curve_tangents_map.copy()
                if is_free_space_available:
                    dij_baseline = planning.Dijkstra(binary_traversable)
                    _, tangent_map_baseline = dij_baseline.plan(greedy_frontier)
                    _, tangent_map_baseline_1 = dij_baseline.plan(greedy_traversable)
                    _, _, _, _, _, _, _, _, tangent_map_edt_1 = edt_with_tangent(binary_traversable, greedy_traversable)
                    _, _, _, _, _, _, _, _, tangent_map_edt_indicator_1 = edt_with_tangent(binary_traversable, greedy_traversable, invalid=label_padded == 0)
                    full_dij_baseline = planning.Dijkstra(full_binary_traversable)
                    _, tangent_map_baseline_2 = full_dij_baseline.plan(full_greedy_traversable)
                    _, _, _, _, _, _, _, _, tangent_map_edt_2 = edt_with_tangent(full_binary_traversable, full_greedy_traversable)
                    _, _, _, _, _, _, _, _, tangent_map_edt_indicator_2 = edt_with_tangent(full_binary_traversable, full_greedy_traversable, invalid=full_label_padded == 0)
            timer.get()

            timer.put("planner")
            if is_ablation:
                planner_straight = planning.Straight(label_mask)
                planner_skeleton = planning.Skeleton(skeleton_binary_traversable, label_mask)
            if is_baseline:
                planner_skeleton_scoring = planning.SkeletonScoring(label_mask.shape[0], label_mask.shape[1], input["res"], input["lidar_range"])
            if is_ablation:
                if is_free_space_available:
                    planner_ogm = self.rrt_planner(1, 1, label_mask, tangent_map_ogm)
                planner_simple = self.rrt_planner(2, 2, label_mask, tangent_map_simple)
                planner_naive = self.rrt_planner(2, 2, label_mask, tangent_map_naive)
                if is_free_space_available:
                    planner_baseline = self.rrt_planner(2, 2, label_mask, tangent_map_baseline)
                    planner_baseline_1 = self.rrt_planner(2, 2, label_mask, tangent_map_baseline_1)
                    planner_edt_1 = self.rrt_planner(2, 2, label_mask, tangent_map_edt_1)
                    planner_edt_indicator_1 = self.rrt_planner(2, 2, label_mask, tangent_map_edt_indicator_1)
                    planner_baseline_2 = self.rrt_planner(2, 2, label_mask, tangent_map_baseline_2)
                    planner_edt_2 = self.rrt_planner(2, 2, label_mask, tangent_map_edt_2)
                    planner_edt_indicator_2 = self.rrt_planner(2, 2, label_mask, tangent_map_edt_indicator_2)
                    planner = self.rrt_planner(1, 1, label_mask, tangent_map)
                planner_pred = self.rrt_planner(1, 1, label_mask, tangent_map_pred)
                planner_pred_consume = self.rrt_planner(2, 1, label_mask, tangent_map_pred)
            planner_pred_consume_invariant = self.rrt_planner(2, 2, label_mask, tangent_map_pred)
            planner_pred_consume_invariant_beta_unifroms15 = self.rrt_planner(2, 2, label_mask, tangent_map_pred_beta_unifroms15)
            planner_pred_consume_invariant_beta_rnt_unifroms15 = self.rnt_planner(2, label_mask, tangent_map_pred_beta_rnt_unifroms15)
            timer.get()

            timer.flush()

            # jobs = []

            # for job in jobs:
            #     job.join()

            # p = multiprocessing.Process(target=worker, args=(i,))
            # jobs.append(p)
            # p.start()

            timer = Timer(True)

            timer.put("greedy_points_straight")
            if is_ablation:
                greedy_points_straight = greedy_point[None]
                traj_straight = planner_straight.plan(greedy_points_straight)
            timer.get()

            dummy_tree = np.full(label_mask.shape, 1)
            dummy_traj = np.full((0, 2), 0.)
            timer.put("greedy_points_ogm")
            if is_ablation:
                greedy_points_ogm = greedy_point[None]
                if is_free_space_available:
                    rrt_tree_ogm, rrt_gain_ogm, rrt_inc_energy_ogm, rrt_inc_gain_ogm, rrt_planned_ogm, rrt_traj_ogm = planner_ogm.plan(self.res, key_meters, greedy_points=greedy_points_ogm)
                else:
                    rrt_tree_ogm = dummy_tree.copy()
                    rrt_traj_ogm = dummy_traj.copy()
            timer.get()

            timer.put("greedy_points_ogm_frontier")
            if is_ablation:
                greedy_points_ogm_frontier = np.transpose(np.nonzero(greedy_frontier))
                if is_free_space_available:
                    rrt_tree_ogm_frontier, rrt_gain_ogm_frontier, rrt_inc_energy_ogm_frontier, rrt_inc_gain_ogm_frontier, rrt_planned_ogm_frontier, rrt_traj_ogm_frontier = planner_ogm.plan(self.res, key_meters, greedy_points=greedy_points_ogm_frontier)
                else:
                    rrt_tree_ogm_frontier = dummy_tree.copy()
                    rrt_traj_ogm_frontier = dummy_traj.copy()
            timer.get()

            timer.put("greedy_points_ogm_skeleton_frontier")
            if is_ablation:
                traj_skeleton = planner_skeleton.plan(skeletonpoint_map, skeleton_cluster_frontier, greedy_points, greedy_tangents)
            if is_baseline:
                SkeletonScoring_root = os.getenv('DATASET_PATH').replace('semantic-kitti', "bev-kitti")
                traj_skeleton_scoring = planner_skeleton_scoring.readfile(frame_id, os.path.join(SkeletonScoring_root, seq_id))
            if is_ablation:
                greedy_points_ogm_skeleton_frontier = np.transpose(np.nonzero((skeleton_cluster_frontier > 0) & self.convertor.matrix_list2map(traj_skeleton)))
                if is_free_space_available:
                    rrt_tree_ogm_skeleton_frontier, rrt_gain_ogm_skeleton_frontier, rrt_inc_energy_ogm_skeleton_frontier, rrt_inc_gain_ogm_skeleton_frontier, rrt_planned_ogm_skeleton_frontier, rrt_traj_ogm_skeleton_frontier = planner_ogm.plan(self.res, key_meters, greedy_points=greedy_points_ogm_skeleton_frontier)
                else:
                    rrt_tree_ogm_skeleton_frontier = dummy_tree.copy()
                    rrt_traj_ogm_skeleton_frontier = dummy_traj.copy()
            timer.get()

            timer.put("several baselines")
            if is_ablation:
                rrt_tree_simple, rrt_gain_simple, rrt_inc_energy_simple, rrt_inc_gain_simple, rrt_planned_simple, rrt_traj_simple = planner_simple.plan(self.res, key_meters)
                rrt_tree_naive, rrt_gain_naive, rrt_inc_energy_naive, rrt_inc_gain_naive, rrt_planned_naive, rrt_traj_naive = planner_naive.plan(self.res, key_meters)
                if is_free_space_available:
                    rrt_tree_baseline, rrt_gain_baseline, rrt_inc_energy_baseline, rrt_inc_gain_baseline, rrt_planned_baseline, rrt_traj_baseline = planner_baseline.plan(self.res, key_meters)
                    rrt_tree_baseline_1, rrt_gain_baseline_1, rrt_inc_energy_baseline_1, rrt_inc_gain_baseline_1, rrt_planned_baseline_1, rrt_traj_baseline_1 = planner_baseline_1.plan(self.res, key_meters)
                    rrt_tree_edt_1, rrt_gain_edt_1, rrt_inc_energy_edt_1, rrt_inc_gain_edt_1, rrt_planned_edt_1, rrt_traj_edt_1 = planner_edt_1.plan(self.res, key_meters)
                    rrt_tree_edt_indicator_1, rrt_gain_edt_indicator_1, rrt_inc_energy_edt_indicator_1, rrt_inc_gain_edt_indicator_1, rrt_planned_edt_indicator_1, rrt_traj_edt_indicator_1 = planner_edt_indicator_1.plan(self.res, key_meters)
                    rrt_tree_baseline_2, rrt_gain_baseline_2, rrt_inc_energy_baseline_2, rrt_inc_gain_baseline_2, rrt_planned_baseline_2, rrt_traj_baseline_2 = planner_baseline_2.plan(self.res, key_meters)
                    rrt_tree_edt_2, rrt_gain_edt_2, rrt_inc_energy_edt_2, rrt_inc_gain_edt_2, rrt_planned_edt_2, rrt_traj_edt_2 = planner_edt_2.plan(self.res, key_meters)
                    rrt_tree_edt_indicator_2, rrt_gain_edt_indicator_2, rrt_inc_energy_edt_indicator_2, rrt_inc_gain_edt_indicator_2, rrt_planned_edt_indicator_2, rrt_traj_edt_indicator_2 = planner_edt_indicator_2.plan(self.res, key_meters)
                    rrt_tree, rrt_gain, rrt_inc_energy, rrt_inc_gain, rrt_planned, rrt_traj = planner.plan(self.res, key_meters)
                else:
                    rrt_tree_baseline = dummy_tree.copy()
                    rrt_traj_baseline = dummy_traj.copy()
                    rrt_tree = dummy_tree.copy()
                    rrt_traj = dummy_traj.copy()
                    rrt_tree_baseline_1 = dummy_tree.copy()
                    rrt_traj_baseline_1 = dummy_traj.copy()
                    rrt_tree_edt_1 = dummy_tree.copy()
                    rrt_traj_edt_1 = dummy_traj.copy()
                    rrt_tree_edt_indicator_1 = dummy_tree.copy()
                    rrt_traj_edt_indicator_1 = dummy_traj.copy()
                    rrt_tree_baseline_2 = dummy_tree.copy()
                    rrt_traj_baseline_2 = dummy_traj.copy()
                    rrt_tree_edt_2 = dummy_tree.copy()
                    rrt_traj_edt_2 = dummy_traj.copy()
                    rrt_tree_edt_indicator_2 = dummy_tree.copy()
                    rrt_traj_edt_indicator_2 = dummy_traj.copy()
            timer.get()

            if is_ablation:
                timer.put("pred")
                rrt_tree_pred, rrt_gain_pred, rrt_inc_energy_pred, rrt_inc_gain_pred, rrt_planned_pred, rrt_traj_pred = planner_pred.plan(self.res, key_meters)
                timer.get()
                timer.put("pred_consume")
                rrt_tree_pred_consume, rrt_gain_pred_consume, rrt_inc_energy_pred_consume, rrt_inc_gain_pred_consume, rrt_planned_pred_consume, rrt_traj_pred_consume = planner_pred_consume.plan(self.res, key_meters)
                timer.get()
            timer.put("pred_consume_invariant")
            rrt_tree_pred_consume_invariant, rrt_gain_pred_consume_invariant, rrt_inc_energy_pred_consume_invariant, rrt_inc_gain_pred_consume_invariant, rrt_planned_pred_consume_invariant, rrt_traj_pred_consume_invariant = planner_pred_consume_invariant.plan(self.res, key_meters)
            timer.get()
            timer.put("pred_consume_invariant_beta")
            rrt_tree_pred_consume_invariant_beta_unifroms15, rrt_gain_pred_consume_invariant_beta_unifroms15, rrt_inc_energy_pred_consume_invariant_beta_unifroms15, rrt_inc_gain_pred_consume_invariant_beta_unifroms15, rrt_planned_pred_consume_invariant_beta_unifroms15, rrt_traj_pred_consume_invariant_beta_unifroms15 = planner_pred_consume_invariant_beta_unifroms15.plan(self.res, key_meters)
            timer.get()
            timer.put("pred_consume_invariant_beta_rnt")
            rrt_tree_pred_consume_invariant_beta_rnt_unifroms15, rrt_gain_pred_consume_invariant_beta_rnt_unifroms15, rrt_inc_energy_pred_consume_invariant_beta_rnt_unifroms15, rrt_inc_gain_pred_consume_invariant_beta_rnt_unifroms15, rrt_planned_pred_consume_invariant_beta_rnt_unifroms15, rrt_traj_pred_consume_invariant_beta_rnt_unifroms15 = planner_pred_consume_invariant_beta_rnt_unifroms15.plan(self.res, key_meters)
            timer.get()
            if is_baseline:
                timer.put("end2end")
                end2end_pointcloud = input['pointcloud']
                end2end_pointcloud = trans_by_switch(end2end_pointcloud)
                end2end_pointcloud[:, 2] += self.end2end.sensor_height
                fall_in_mask = (end2end_pointcloud[:, 0] >= self.end2end.lidar_range['Left']) & (end2end_pointcloud[:, 0] < self.end2end.lidar_range['Right']) & (end2end_pointcloud[:, 1] > self.end2end.lidar_range['Back']) & (end2end_pointcloud[:, 1] <= self.end2end.lidar_range['Front'])
                end2end_pointcloud = end2end_pointcloud[fall_in_mask]
                end2end_bev_map = bev.rgb_map(end2end_pointcloud[:, :4], self.end2end.lidar_range_np, self.end2end.res)
                end2end_bev_map = end2end_bev_map.reshape((self.end2end.img_h, self.end2end.img_w, -1)).copy()
                rescale = 1.0
                thickness = int(rescale * self.end2end.road_width / self.end2end.res)
                img_hmi = np.zeros((int(self.end2end.img_h * rescale), int(self.end2end.img_w * rescale), 1)).astype(np.uint8)
                traj_hmi_all = self.convertor.matrix2car(eline_points)
                for i in range(eline_points.shape[0] - 1):
                    x1, y1 = self.end2end.car2img(traj_hmi_all[i][:2])
                    x2, y2 = self.end2end.car2img(traj_hmi_all[i+1][:2])
                    x1 = int(x1 * rescale)
                    y1 = int(y1 * rescale)
                    x2 = int(x2 * rescale)
                    y2 = int(y2 * rescale)
                    cv2.line(img_hmi, (x1, y1), (x2, y2), (255,255,255), thickness)
                # plt.imshow(end2end_bev_map)
                # plt.show()
                # plt.imshow(img_hmi)
                # plt.show()
                end2end_outputs = self.end2end.inference({
                    "lidar_bev": end2end_bev_map.transpose(2, 0, 1)[None],
                    "img_hmi": ToTensor()(img_hmi)[None],
                })
                end2end_traj_fly = end2end_outputs[0]
                end2end_traj_fly = self.convertor.car2matrix(end2end_traj_fly)
                # print("end2end diff", np.linalg.norm(end2end_traj_base - end2end_traj_fly, axis=-1).sum())
                # plt.imshow(np.concatenate([self.convertor.matrix_list2map(end2end_traj_base), self.convertor.matrix_list2map(end2end_traj_fly)]))
                # plt.show()
                end2end_traj = end2end_traj_fly.copy()
                timer.get()

            timer.flush()

            timer.put("eval")
            metrics = []
            commons = []
            cols = []
            waypoints = []
            trajs = []
            failure = False
            for key_meters in eval_meters:

                handler = planning.PathHandler(label_mask.shape[0], label_mask.shape[1], self.res, key_meters)
                if is_ablation:
                    handled_traj_straight = handler.handle(traj_straight)
                    handled_traj_skeleton = handler.handle(traj_skeleton)
                if is_baseline:
                    handled_traj_skeleton_scoring = handler.handle(traj_skeleton_scoring)
                if is_ablation:
                    handled_traj_ogm = handler.handle(rrt_traj_ogm)
                    handled_traj_ogm_frontier = handler.handle(rrt_traj_ogm_frontier)
                    handled_traj_ogm_skeleton_frontier = handler.handle(rrt_traj_ogm_skeleton_frontier)
                    handled_traj_simple = handler.handle(rrt_traj_simple)
                    handled_traj_naive = handler.handle(rrt_traj_naive)
                    handled_traj_baseline = handler.handle(rrt_traj_baseline)
                    handled_traj_baseline_1 = handler.handle(rrt_traj_baseline_1)
                    handled_traj_edt_1 = handler.handle(rrt_traj_edt_1)
                    handled_traj_edt_indicator_1 = handler.handle(rrt_traj_edt_indicator_1)
                    handled_traj_baseline_2 = handler.handle(rrt_traj_baseline_2)
                    handled_traj_edt_2 = handler.handle(rrt_traj_edt_2)
                    handled_traj_edt_indicator_2 = handler.handle(rrt_traj_edt_indicator_2)
                    handled_traj = handler.handle(rrt_traj)
                    handled_traj_pred = handler.handle(rrt_traj_pred)
                    handled_traj_pred_consume = handler.handle(rrt_traj_pred_consume)
                handled_traj_pred_consume_invariant = handler.handle(rrt_traj_pred_consume_invariant)
                handled_traj_pred_consume_invariant_beta_unifroms15 = handler.handle(rrt_traj_pred_consume_invariant_beta_unifroms15)
                handled_traj_pred_consume_invariant_beta_rnt_unifroms15 = handler.handle(rrt_traj_pred_consume_invariant_beta_rnt_unifroms15)
                if is_baseline:
                    handled_traj_end2end = handler.handle(end2end_traj)

                if is_ablation:
                    traj_diff_straight = self.traj_diff(target_traj, target_points, (handled_traj_straight), key_meters)
                    traj_diff_skeleton = self.traj_diff(target_traj, target_points, (handled_traj_skeleton), key_meters)
                if is_baseline:
                    traj_diff_skeleton_scoring = self.traj_diff(target_traj, target_points, (handled_traj_skeleton_scoring), key_meters)
                if is_ablation:
                    traj_diff_ogm = self.traj_diff(target_traj, target_points, (handled_traj_ogm), key_meters)
                    traj_diff_ogm_frontier = self.traj_diff(target_traj, target_points, (handled_traj_ogm_frontier), key_meters)
                    traj_diff_ogm_skeleton_frontier = self.traj_diff(target_traj, target_points, (handled_traj_ogm_skeleton_frontier), key_meters)
                    traj_diff_simple = self.traj_diff(target_traj, target_points, (handled_traj_simple), key_meters)
                    traj_diff_naive = self.traj_diff(target_traj, target_points, (handled_traj_naive), key_meters)
                    traj_diff_baseline = self.traj_diff(target_traj, target_points, (handled_traj_baseline), key_meters)
                    traj_diff_baseline_1 = self.traj_diff(target_traj, target_points, (handled_traj_baseline_1), key_meters)
                    traj_diff_edt_1 = self.traj_diff(target_traj, target_points, (handled_traj_edt_1), key_meters)
                    traj_diff_edt_indicator_1 = self.traj_diff(target_traj, target_points, (handled_traj_edt_indicator_1), key_meters)
                    traj_diff_baseline_2 = self.traj_diff(target_traj, target_points, (handled_traj_baseline_2), key_meters)
                    traj_diff_edt_2 = self.traj_diff(target_traj, target_points, (handled_traj_edt_2), key_meters)
                    traj_diff_edt_indicator_2 = self.traj_diff(target_traj, target_points, (handled_traj_edt_indicator_2), key_meters)
                    traj_diff = self.traj_diff(target_traj, target_points, (handled_traj), key_meters)
                    traj_diff_pred = self.traj_diff(target_traj, target_points, (handled_traj_pred), key_meters)
                    traj_diff_pred_consume = self.traj_diff(target_traj, target_points, (handled_traj_pred_consume), key_meters)
                traj_diff_pred_consume_invariant = self.traj_diff(target_traj, target_points, (handled_traj_pred_consume_invariant), key_meters)
                traj_diff_pred_consume_invariant_beta_unifroms15 = self.traj_diff(target_traj, target_points, (handled_traj_pred_consume_invariant_beta_unifroms15), key_meters)
                traj_diff_pred_consume_invariant_beta_rnt_unifroms15 = self.traj_diff(target_traj, target_points, (handled_traj_pred_consume_invariant_beta_rnt_unifroms15), key_meters)
                if is_baseline:
                    traj_diff_end2end = self.traj_diff(target_traj, target_points, (handled_traj_end2end), key_meters)
                key_diffs = {
                        "pred_consume_invariant": traj_diff_pred_consume_invariant,
                        "pred_consume_invariant_beta_unifroms15": traj_diff_pred_consume_invariant_beta_unifroms15,
                        "pred_consume_invariant_beta_rnt_unifroms15": traj_diff_pred_consume_invariant_beta_rnt_unifroms15,
                }
                if is_baseline:
                    key_diffs.update({
                        "skeleton_scoring": traj_diff_skeleton_scoring,
                        "end2end": traj_diff_end2end,
                    })
                if is_ablation:
                    key_diffs.update({
                        "straight": traj_diff_straight,
                        "skeleton": traj_diff_skeleton,
                        "ogm": traj_diff_ogm,
                        "ogm_frontier": traj_diff_ogm_frontier,
                        "ogm_skeleton_frontier": traj_diff_ogm_skeleton_frontier,
                        "simple": traj_diff_simple,
                        "naive": traj_diff_naive,
                        "baseline": traj_diff_baseline,
                        "baseline_1": traj_diff_baseline_1,
                        "edt_1": traj_diff_edt_1,
                        "edt_indicator_1": traj_diff_edt_indicator_1,
                        "baseline_2": traj_diff_baseline_2,
                        "edt_2": traj_diff_edt_2,
                        "edt_indicator_2": traj_diff_edt_indicator_2,
                        "sup": traj_diff,
                        "pred": traj_diff_pred,
                        "pred_consume": traj_diff_pred_consume,
                    })

                if is_ablation:
                    traj_dict_straight, pred_interp_straight, target_interp_straight = self.traj_dict(target_traj, target_points, (handled_traj_straight), key_meters)
                    traj_dict_skeleton, pred_interp_skeleton, target_interp_skeleton = self.traj_dict(target_traj, target_points, (handled_traj_skeleton), key_meters)
                if is_baseline:
                    traj_dict_skeleton_scoring, pred_interp_skeleton_scoring, target_interp_skeleton_scoring = self.traj_dict(target_traj, target_points, (handled_traj_skeleton_scoring), key_meters)
                if is_ablation:
                    traj_dict_ogm, pred_interp_ogm, target_interp_ogm = self.traj_dict(target_traj, target_points, (handled_traj_ogm), key_meters)
                    traj_dict_ogm_frontier, pred_interp_ogm_frontier, target_interp_ogm_frontier = self.traj_dict(target_traj, target_points, (handled_traj_ogm_frontier), key_meters)
                    traj_dict_ogm_skeleton_frontier, pred_interp_ogm_skeleton_frontier, target_interp_ogm_skeleton_frontier = self.traj_dict(target_traj, target_points, (handled_traj_ogm_skeleton_frontier), key_meters)
                    traj_dict_simple, pred_interp_simple, target_interp_simple = self.traj_dict(target_traj, target_points, (handled_traj_simple), key_meters)
                    traj_dict_naive, pred_interp_naive, target_interp_naive = self.traj_dict(target_traj, target_points, (handled_traj_naive), key_meters)
                    traj_dict_baseline, pred_interp_baseline, target_interp_baseline = self.traj_dict(target_traj, target_points, (handled_traj_baseline), key_meters)
                    traj_dict_baseline_1, pred_interp_baseline_1, target_interp_baseline_1 = self.traj_dict(target_traj, target_points, (handled_traj_baseline_1), key_meters)
                    traj_dict_edt_1, pred_interp_edt_1, target_interp_edt_1 = self.traj_dict(target_traj, target_points, (handled_traj_edt_1), key_meters)
                    traj_dict_edt_indicator_1, pred_interp_edt_indicator_1, target_interp_edt_indicator_1 = self.traj_dict(target_traj, target_points, (handled_traj_edt_indicator_1), key_meters)
                    traj_dict_baseline_2, pred_interp_baseline_2, target_interp_baseline_2 = self.traj_dict(target_traj, target_points, (handled_traj_baseline_2), key_meters)
                    traj_dict_edt_2, pred_interp_edt_2, target_interp_edt_2 = self.traj_dict(target_traj, target_points, (handled_traj_edt_2), key_meters)
                    traj_dict_edt_indicator_2, pred_interp_edt_indicator_2, target_interp_edt_indicator_2 = self.traj_dict(target_traj, target_points, (handled_traj_edt_indicator_2), key_meters)
                    traj_dict, pred_interp, target_interp = self.traj_dict(target_traj, target_points, (handled_traj), key_meters)
                    traj_dict_pred, pred_interp_pred, target_interp_pred = self.traj_dict(target_traj, target_points, (handled_traj_pred), key_meters)
                    traj_dict_pred_consume, pred_interp_pred_consume, target_interp_pred_consume = self.traj_dict(target_traj, target_points, (handled_traj_pred_consume), key_meters)
                traj_dict_pred_consume_invariant, pred_interp_pred_consume_invariant, target_interp_pred_consume_invariant = self.traj_dict(target_traj, target_points, (handled_traj_pred_consume_invariant), key_meters)
                traj_dict_pred_consume_invariant_beta_unifroms15, pred_interp_pred_consume_invariant_beta_unifroms15, target_interp_pred_consume_invariant_beta_unifroms15 = self.traj_dict(target_traj, target_points, (handled_traj_pred_consume_invariant_beta_unifroms15), key_meters)
                traj_dict_pred_consume_invariant_beta_rnt_unifroms15, pred_interp_pred_consume_invariant_beta_rnt_unifroms15, target_interp_pred_consume_invariant_beta_rnt_unifroms15 = self.traj_dict(target_traj, target_points, (handled_traj_pred_consume_invariant_beta_rnt_unifroms15), key_meters)
                if is_baseline:
                    traj_dict_end2end, pred_interp_end2end, target_interp_end2end = self.traj_dict(target_traj, target_points, (handled_traj_end2end), key_meters)
                key_metrics = {
                        "pred_consume_invariant": traj_dict_pred_consume_invariant,
                        "pred_consume_invariant_beta_unifroms15": traj_dict_pred_consume_invariant_beta_unifroms15,
                        "pred_consume_invariant_beta_rnt_unifroms15": traj_dict_pred_consume_invariant_beta_rnt_unifroms15,
                }
                if is_baseline:
                    key_metrics.update({
                        "skeleton_scoring": traj_dict_skeleton_scoring,
                        "end2end": traj_dict_end2end,
                    })
                if is_ablation:
                    key_metrics.update({
                        "straight": traj_dict_straight,
                        "skeleton": traj_dict_skeleton,
                        "ogm": traj_dict_ogm,
                        "ogm_frontier": traj_dict_ogm_frontier,
                        "ogm_skeleton_frontier": traj_dict_ogm_skeleton_frontier,
                        "simple": traj_dict_simple,
                        "naive": traj_dict_naive,
                        "baseline": traj_dict_baseline,
                        "baseline_1": traj_dict_baseline_1,
                        "edt_1": traj_dict_edt_1,
                        "edt_indicator_1": traj_dict_edt_indicator_1,
                        "baseline_2": traj_dict_baseline_2,
                        "edt_2": traj_dict_edt_2,
                        "edt_indicator_2": traj_dict_edt_indicator_2,
                        "sup": traj_dict,
                        "pred": traj_dict_pred,
                        "pred_consume": traj_dict_pred_consume,
                    })
                key_trajs = {
                        "pred_consume_invariant": {f"traj_{key_meters}m": np.concatenate([pred_interp_pred_consume_invariant, target_interp_pred_consume_invariant], axis=-1)},
                        "pred_consume_invariant_beta_unifroms15": {f"traj_{key_meters}m": np.concatenate([pred_interp_pred_consume_invariant_beta_unifroms15, target_interp_pred_consume_invariant_beta_unifroms15], axis=-1)},
                        "pred_consume_invariant_beta_rnt_unifroms15": {f"traj_{key_meters}m": np.concatenate([pred_interp_pred_consume_invariant_beta_rnt_unifroms15, target_interp_pred_consume_invariant_beta_rnt_unifroms15], axis=-1)},
                }
                if is_baseline:
                    key_trajs.update({
                        "skeleton_scoring": {f"traj_{key_meters}m": np.concatenate([pred_interp_skeleton_scoring, target_interp_skeleton_scoring], axis=-1)},
                        "end2end": {f"traj_{key_meters}m": np.concatenate([pred_interp_end2end, target_interp_end2end], axis=-1)},
                    })
                if is_ablation:
                    key_trajs.update({
                        "straight": {f"traj_{key_meters}m": np.concatenate([pred_interp_straight, target_interp_straight], axis=-1)},
                        "skeleton": {f"traj_{key_meters}m": np.concatenate([pred_interp_skeleton, target_interp_skeleton], axis=-1)},
                        "ogm": {f"traj_{key_meters}m": np.concatenate([pred_interp_ogm, target_interp_ogm], axis=-1)},
                        "ogm_frontier": {f"traj_{key_meters}m": np.concatenate([pred_interp_ogm_frontier, target_interp_ogm_frontier], axis=-1)},
                        "ogm_skeleton_frontier": {f"traj_{key_meters}m": np.concatenate([pred_interp_ogm_skeleton_frontier, target_interp_ogm_skeleton_frontier], axis=-1)},
                        "simple": {f"traj_{key_meters}m": np.concatenate([pred_interp_simple, target_interp_simple], axis=-1)},
                        "naive": {f"traj_{key_meters}m": np.concatenate([pred_interp_naive, target_interp_naive], axis=-1)},
                        "baseline": {f"traj_{key_meters}m": np.concatenate([pred_interp_baseline, target_interp_baseline], axis=-1)},
                        "baseline_1": {f"traj_{key_meters}m": np.concatenate([pred_interp_baseline_1, target_interp_baseline_1], axis=-1)},
                        "edt_1": {f"traj_{key_meters}m": np.concatenate([pred_interp_edt_1, target_interp_edt_1], axis=-1)},
                        "edt_indicator_1": {f"traj_{key_meters}m": np.concatenate([pred_interp_edt_indicator_1, target_interp_edt_indicator_1], axis=-1)},
                        "baseline_2": {f"traj_{key_meters}m": np.concatenate([pred_interp_baseline_2, target_interp_baseline_2], axis=-1)},
                        "edt_2": {f"traj_{key_meters}m": np.concatenate([pred_interp_edt_2, target_interp_edt_2], axis=-1)},
                        "edt_indicator_2": {f"traj_{key_meters}m": np.concatenate([pred_interp_edt_indicator_2, target_interp_edt_indicator_2], axis=-1)},
                        "sup": {f"traj_{key_meters}m": np.concatenate([pred_interp, target_interp], axis=-1)},
                        "pred": {f"traj_{key_meters}m": np.concatenate([pred_interp_pred, target_interp_pred], axis=-1)},
                        "pred_consume": {f"traj_{key_meters}m": np.concatenate([pred_interp_pred_consume, target_interp_pred_consume], axis=-1)},
                    })

                if is_ablation:
                    traj_coverage_straight = self.traj_coverage(target_traj, target_points, (handled_traj_straight), key_meters)
                    traj_coverage_skeleton = self.traj_coverage(target_traj, target_points, (handled_traj_skeleton), key_meters)
                if is_baseline:
                    traj_coverage_skeleton_scoring = self.traj_coverage(target_traj, target_points, (handled_traj_skeleton_scoring), key_meters)
                if is_ablation:
                    traj_coverage_ogm = self.traj_coverage(target_traj, target_points, (handled_traj_ogm), key_meters)
                    traj_coverage_ogm_frontier = self.traj_coverage(target_traj, target_points, (handled_traj_ogm_frontier), key_meters)
                    traj_coverage_ogm_skeleton_frontier = self.traj_coverage(target_traj, target_points, (handled_traj_ogm_skeleton_frontier), key_meters)
                    traj_coverage_simple = self.traj_coverage(target_traj, target_points, (handled_traj_simple), key_meters)
                    traj_coverage_naive = self.traj_coverage(target_traj, target_points, (handled_traj_naive), key_meters)
                    traj_coverage_baseline = self.traj_coverage(target_traj, target_points, (handled_traj_baseline), key_meters)
                    traj_coverage_baseline_1 = self.traj_coverage(target_traj, target_points, (handled_traj_baseline_1), key_meters)
                    traj_coverage_edt_1 = self.traj_coverage(target_traj, target_points, (handled_traj_edt_1), key_meters)
                    traj_coverage_edt_indicator_1 = self.traj_coverage(target_traj, target_points, (handled_traj_edt_indicator_1), key_meters)
                    traj_coverage_baseline_2 = self.traj_coverage(target_traj, target_points, (handled_traj_baseline_2), key_meters)
                    traj_coverage_edt_2 = self.traj_coverage(target_traj, target_points, (handled_traj_edt_2), key_meters)
                    traj_coverage_edt_indicator_2 = self.traj_coverage(target_traj, target_points, (handled_traj_edt_indicator_2), key_meters)
                    traj_coverage = self.traj_coverage(target_traj, target_points, (handled_traj), key_meters)
                    traj_coverage_pred = self.traj_coverage(target_traj, target_points, (handled_traj_pred), key_meters)
                    traj_coverage_pred_consume = self.traj_coverage(target_traj, target_points, (handled_traj_pred_consume), key_meters)
                traj_coverage_pred_consume_invariant = self.traj_coverage(target_traj, target_points, (handled_traj_pred_consume_invariant), key_meters)
                traj_coverage_pred_consume_invariant_beta_unifroms15 = self.traj_coverage(target_traj, target_points, (handled_traj_pred_consume_invariant_beta_unifroms15), key_meters)
                traj_coverage_pred_consume_invariant_beta_rnt_unifroms15 = self.traj_coverage(target_traj, target_points, (handled_traj_pred_consume_invariant_beta_rnt_unifroms15), key_meters)
                if is_baseline:
                    traj_coverage_end2end = self.traj_coverage(target_traj, target_points, (handled_traj_end2end), key_meters)
                key_coverages = {
                        "pred_consume_invariant": traj_coverage_pred_consume_invariant,
                        "pred_consume_invariant_beta_unifroms15": traj_coverage_pred_consume_invariant_beta_unifroms15,
                        "pred_consume_invariant_beta_rnt_unifroms15": traj_coverage_pred_consume_invariant_beta_rnt_unifroms15,
                }
                if is_baseline:
                    key_coverages.update({
                        "skeleton_scoring": traj_coverage_skeleton_scoring,
                        "end2end": traj_coverage_end2end,
                    })
                if is_ablation:
                    key_coverages.update({
                        "straight": traj_coverage_straight,
                        "skeleton": traj_coverage_skeleton,
                        "ogm": traj_coverage_ogm,
                        "ogm_frontier": traj_coverage_ogm_frontier,
                        "ogm_skeleton_frontier": traj_coverage_ogm_skeleton_frontier,
                        "simple": traj_coverage_simple,
                        "naive": traj_coverage_naive,
                        "baseline": traj_coverage_baseline,
                        "baseline_1": traj_coverage_baseline_1,
                        "edt_1": traj_coverage_edt_1,
                        "edt_indicator_1": traj_coverage_edt_indicator_1,
                        "baseline_2": traj_coverage_baseline_2,
                        "edt_2": traj_coverage_edt_2,
                        "edt_indicator_2": traj_coverage_edt_indicator_2,
                        "sup": traj_coverage,
                        "pred": traj_coverage_pred,
                        "pred_consume": traj_coverage_pred_consume,
                    })

                if is_ablation:
                    traj_coverage_interp_straight = self.traj_coverage_interp(pred_interp_straight, target_interp_straight, key_meters)
                    traj_coverage_interp_skeleton = self.traj_coverage_interp(pred_interp_skeleton, target_interp_skeleton, key_meters)
                if is_baseline:
                    traj_coverage_interp_skeleton_scoring = self.traj_coverage_interp(pred_interp_skeleton_scoring, target_interp_skeleton_scoring, key_meters)
                if is_ablation:
                    traj_coverage_interp_ogm = self.traj_coverage_interp(pred_interp_ogm, target_interp_ogm, key_meters)
                    traj_coverage_interp_ogm_frontier = self.traj_coverage_interp(pred_interp_ogm_frontier, target_interp_ogm_frontier, key_meters)
                    traj_coverage_interp_ogm_skeleton_frontier = self.traj_coverage_interp(pred_interp_ogm_skeleton_frontier, target_interp_ogm_skeleton_frontier, key_meters)
                    traj_coverage_interp_simple = self.traj_coverage_interp(pred_interp_simple, target_interp_simple, key_meters)
                    traj_coverage_interp_naive = self.traj_coverage_interp(pred_interp_naive, target_interp_naive, key_meters)
                    traj_coverage_interp_baseline = self.traj_coverage_interp(pred_interp_baseline, target_interp_baseline, key_meters)
                    traj_coverage_interp_baseline_1 = self.traj_coverage_interp(pred_interp_baseline_1, target_interp_baseline_1, key_meters)
                    traj_coverage_interp_edt_1 = self.traj_coverage_interp(pred_interp_edt_1, target_interp_edt_1, key_meters)
                    traj_coverage_interp_edt_indicator_1 = self.traj_coverage_interp(pred_interp_edt_indicator_1, target_interp_edt_indicator_1, key_meters)
                    traj_coverage_interp_baseline_2 = self.traj_coverage_interp(pred_interp_baseline_2, target_interp_baseline_2, key_meters)
                    traj_coverage_interp_edt_2 = self.traj_coverage_interp(pred_interp_edt_2, target_interp_edt_2, key_meters)
                    traj_coverage_interp_edt_indicator_2 = self.traj_coverage_interp(pred_interp_edt_indicator_2, target_interp_edt_indicator_2, key_meters)
                    traj_coverage_interp = self.traj_coverage_interp(pred_interp, target_interp, key_meters)
                    traj_coverage_interp_pred = self.traj_coverage_interp(pred_interp_pred, target_interp_pred, key_meters)
                    traj_coverage_interp_pred_consume = self.traj_coverage_interp(pred_interp_pred_consume, target_interp_pred_consume, key_meters)
                traj_coverage_interp_pred_consume_invariant = self.traj_coverage_interp(pred_interp_pred_consume_invariant, target_interp_pred_consume_invariant, key_meters)
                traj_coverage_interp_pred_consume_invariant_beta_unifroms15 = self.traj_coverage_interp(pred_interp_pred_consume_invariant_beta_unifroms15, target_interp_pred_consume_invariant_beta_unifroms15, key_meters)
                traj_coverage_interp_pred_consume_invariant_beta_rnt_unifroms15 = self.traj_coverage_interp(pred_interp_pred_consume_invariant_beta_rnt_unifroms15, target_interp_pred_consume_invariant_beta_rnt_unifroms15, key_meters)
                if is_baseline:
                    traj_coverage_interp_end2end = self.traj_coverage_interp(pred_interp_end2end, target_interp_end2end, key_meters)
                key_coverage_interps = {
                    "pred_consume_invariant": traj_coverage_interp_pred_consume_invariant,
                    "pred_consume_invariant_beta_unifroms15": traj_coverage_interp_pred_consume_invariant_beta_unifroms15,
                    "pred_consume_invariant_beta_rnt_unifroms15": traj_coverage_interp_pred_consume_invariant_beta_rnt_unifroms15,
                }
                if is_baseline:
                    key_coverage_interps.update({
                        "skeleton_scoring": traj_coverage_interp_skeleton_scoring,
                        "end2end": traj_coverage_interp_end2end,
                    })
                if is_ablation:
                    key_coverage_interps.update({
                        "straight": traj_coverage_interp_straight,
                        "skeleton": traj_coverage_interp_skeleton,
                        "ogm": traj_coverage_interp_ogm,
                        "ogm_frontier": traj_coverage_interp_ogm_frontier,
                        "ogm_skeleton_frontier": traj_coverage_interp_ogm_skeleton_frontier,
                        "simple": traj_coverage_interp_simple,
                        "naive": traj_coverage_interp_naive,
                        "baseline": traj_coverage_interp_baseline,
                        "baseline_1": traj_coverage_interp_baseline_1,
                        "edt_1": traj_coverage_interp_edt_1,
                        "edt_indicator_1": traj_coverage_interp_edt_indicator_1,
                        "baseline_2": traj_coverage_interp_baseline_2,
                        "edt_2": traj_coverage_interp_edt_2,
                        "edt_indicator_2": traj_coverage_interp_edt_indicator_2,
                        "sup": traj_coverage_interp,
                        "pred": traj_coverage_interp_pred,
                        "pred_consume": traj_coverage_interp_pred_consume,
                    })

                if is_ablation:
                    waypoint_diff_straight, waypoint_straight_car, _, waypoint_straight, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_straight), key_meters)
                    waypoint_diff_skeleton, waypoint_skeleton_car, _, waypoint_skeleton, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_skeleton), key_meters)
                if is_baseline:
                    waypoint_diff_skeleton_scoring, waypoint_skeleton_scoring_car, waypoint_skeleton_scoring_target_car, waypoint_skeleton_scoring, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_skeleton_scoring), key_meters)
                if is_ablation:
                    waypoint_diff_ogm, waypoint_ogm_car, _, waypoint_ogm, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_ogm), key_meters)
                    waypoint_diff_ogm_frontier, waypoint_ogm_frontier_car, _, waypoint_ogm_frontier, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_ogm_frontier), key_meters)
                    waypoint_diff_ogm_skeleton_frontier, waypoint_ogm_skeleton_frontier_car, _, waypoint_ogm_skeleton_frontier, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_ogm_skeleton_frontier), key_meters)
                    waypoint_diff_simple, waypoint_simple_car, _, waypoint_simple, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_simple), key_meters)
                    waypoint_diff_naive, waypoint_naive_car, _, waypoint_naive, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_naive), key_meters)
                    waypoint_diff_baseline, waypoint_baseline_car, _, waypoint_baseline, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_baseline), key_meters)
                    waypoint_diff_baseline_1, waypoint_baseline_1_car, _, waypoint_baseline_1, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_baseline_1), key_meters)
                    waypoint_diff_edt_1, waypoint_edt_1_car, _, waypoint_edt_1, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_edt_1), key_meters)
                    waypoint_diff_edt_indicator_1, waypoint_edt_indicator_1_car, _, waypoint_edt_indicator_1, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_edt_indicator_1), key_meters)
                    waypoint_diff_baseline_2, waypoint_baseline_2_car, _, waypoint_baseline_2, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_baseline_2), key_meters)
                    waypoint_diff_edt_2, waypoint_edt_2_car, _, waypoint_edt_2, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_edt_2), key_meters)
                    waypoint_diff_edt_indicator_2, waypoint_edt_indicator_2_car, _, waypoint_edt_indicator_2, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_edt_indicator_2), key_meters)
                    waypoint_diff, waypoint_car, waypoint_target_car, waypoint, waypoint_target = self.waypoint_diff(target_traj, target_points, (handled_traj), key_meters)
                    waypoint_diff_pred, waypoint_pred_car, _, waypoint_pred, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_pred), key_meters)
                    waypoint_diff_pred_consume, waypoint_pred_consume_car, _, waypoint_pred_consume, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_pred_consume), key_meters)
                waypoint_diff_pred_consume_invariant, waypoint_pred_consume_invariant_car, waypoint_pred_consume_invariant_target_car, waypoint_pred_consume_invariant, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_pred_consume_invariant), key_meters)
                waypoint_diff_pred_consume_invariant_beta_unifroms15, waypoint_pred_consume_invariant_beta_unifroms15_car, waypoint_pred_consume_invariant_beta_unifroms15_target_car, waypoint_pred_consume_invariant_beta_unifroms15, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_pred_consume_invariant_beta_unifroms15), key_meters)
                waypoint_diff_pred_consume_invariant_beta_rnt_unifroms15, waypoint_pred_consume_invariant_beta_rnt_unifroms15_car, waypoint_pred_consume_invariant_beta_rnt_unifroms15_target_car, waypoint_pred_consume_invariant_beta_rnt_unifroms15, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_pred_consume_invariant_beta_rnt_unifroms15), key_meters)
                if is_baseline:
                    waypoint_diff_end2end, waypoint_end2end_car, waypoint_end2end_target_car, waypoint_end2end, _ = self.waypoint_diff(target_traj, target_points, (handled_traj_end2end), key_meters)
                key_waypoint_diffs = {
                    "pred_consume_invariant": waypoint_diff_pred_consume_invariant,
                    "pred_consume_invariant_beta_unifroms15": waypoint_diff_pred_consume_invariant_beta_unifroms15,
                    "pred_consume_invariant_beta_rnt_unifroms15": waypoint_diff_pred_consume_invariant_beta_rnt_unifroms15,
                }
                if is_baseline:
                    key_waypoint_diffs.update({
                        "skeleton_scoring": waypoint_diff_skeleton_scoring,
                        "end2end": waypoint_diff_end2end,
                    })
                if is_ablation:
                    key_waypoint_diffs.update({
                        "straight": waypoint_diff_straight,
                        "skeleton": waypoint_diff_skeleton,
                        "ogm": waypoint_diff_ogm,
                        "ogm_frontier": waypoint_diff_ogm_frontier,
                        "ogm_skeleton_frontier": waypoint_diff_ogm_skeleton_frontier,
                        "simple": waypoint_diff_simple,
                        "naive": waypoint_diff_naive,
                        "baseline": waypoint_diff_baseline,
                        "baseline_1": waypoint_diff_baseline_1,
                        "edt_1": waypoint_diff_edt_1,
                        "edt_indicator_1": waypoint_diff_edt_indicator_1,
                        "baseline_2": waypoint_diff_baseline_2,
                        "edt_2": waypoint_diff_edt_2,
                        "edt_indicator_2": waypoint_diff_edt_indicator_2,
                        "sup": waypoint_diff,
                        "pred": waypoint_diff_pred,
                        "pred_consume": waypoint_diff_pred_consume,
                    })
                key_waypoints = {
                    "pred_consume_invariant": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_pred_consume_invariant[-1], waypoint_pred_consume_invariant_car, waypoint_pred_consume_invariant_target_car])},
                    "pred_consume_invariant_beta_unifroms15": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_pred_consume_invariant_beta_unifroms15[-1], waypoint_pred_consume_invariant_beta_unifroms15_car, waypoint_pred_consume_invariant_beta_unifroms15_target_car])},
                    "pred_consume_invariant_beta_rnt_unifroms15": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_pred_consume_invariant_beta_rnt_unifroms15[-1], waypoint_pred_consume_invariant_beta_rnt_unifroms15_car, waypoint_pred_consume_invariant_beta_rnt_unifroms15_target_car])},
                }
                if is_baseline:
                    key_waypoints.update({
                        "skeleton_scoring": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_skeleton_scoring[-1], waypoint_skeleton_scoring_car, waypoint_skeleton_scoring_target_car])},
                        "end2end": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_end2end[-1], waypoint_end2end_car, waypoint_end2end_target_car])},
                    })
                if is_ablation:
                    key_waypoints.update({
                        "straight": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_straight[-1], waypoint_straight_car, waypoint_target_car])},
                        "skeleton": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_skeleton[-1], waypoint_skeleton_car, waypoint_target_car])},
                        "ogm": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_ogm[-1], waypoint_ogm_car, waypoint_target_car])},
                        "ogm_frontier": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_ogm_frontier[-1], waypoint_ogm_frontier_car, waypoint_target_car])},
                        "ogm_skeleton_frontier": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_ogm_skeleton_frontier[-1], waypoint_ogm_skeleton_frontier_car, waypoint_target_car])},
                        "simple": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_simple[-1], waypoint_simple_car, waypoint_target_car])},
                        "naive": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_naive[-1], waypoint_naive_car, waypoint_target_car])},
                        "baseline": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_baseline[-1], waypoint_baseline_car, waypoint_target_car])},
                        "baseline_1": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_baseline_1[-1], waypoint_baseline_1_car, waypoint_target_car])},
                        "edt_1": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_edt_1[-1], waypoint_edt_1_car, waypoint_target_car])},
                        "edt_indicator_1": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_edt_indicator_1[-1], waypoint_edt_indicator_1_car, waypoint_target_car])},
                        "baseline_2": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_baseline_2[-1], waypoint_baseline_2_car, waypoint_target_car])},
                        "edt_2": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_edt_2[-1], waypoint_edt_2_car, waypoint_target_car])},
                        "edt_indicator_2": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_edt_indicator_2[-1], waypoint_edt_indicator_2_car, waypoint_target_car])},
                        "sup": {f"waypoint_{key_meters}m": np.concatenate([pred_interp[-1], waypoint_car, waypoint_target_car])},
                        "pred": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_pred[-1], waypoint_pred_car, waypoint_target_car])},
                        "pred_consume": {f"waypoint_{key_meters}m": np.concatenate([pred_interp_pred_consume[-1], waypoint_pred_consume_car, waypoint_target_car])},
                    })

                metrics.extend([key_metrics])
                commons.extend([key_diffs, key_coverages, key_coverage_interps, key_waypoint_diffs])
                cols.extend([
                    f"ADE_{key_meters}m", 
                    f"FDE_{key_meters}m", 
                    f"HitRate_{key_meters}m", 
                    f"HD_{key_meters}m",
                    f"diff_{key_meters}m", 
                    f"cover_pre_{key_meters}m",
                    f"cover_rec_{key_meters}m",
                    f"cover_F1_{key_meters}m",
                    f"cover_interp_rec_{key_meters}m",
                    f"cover_interp_IoU_{key_meters}m",
                    f"waypoint_{key_meters}m", 
                ])
                waypoints.extend([key_waypoints])
                trajs.extend([key_trajs])

            metas = {
                    "seq_id": seq_id,
                    "frame_id": frame_id,
                    "pose": pose,
                }
            metrics = self.merge_dict(metrics)
            commons = self.merge_dict(commons)
            waypoints = self.merge_dict(waypoints)
            trajs = self.merge_dict(trajs)
            self.cols = cols
            self.rows = []
            if is_ablation:
                self.rows.append("straight")
                self.rows.append("skeleton")
            if is_baseline:
                self.rows.append("skeleton_scoring")
            if is_ablation:
                self.rows.append("ogm")
                self.rows.append("ogm_frontier")
                self.rows.append("ogm_skeleton_frontier")
                self.rows.append("simple")
                self.rows.append("naive")
                self.rows.append("baseline")
                self.rows.append("baseline_1")
                self.rows.append("edt_1")
                self.rows.append("edt_indicator_1")
                self.rows.append("baseline_2")
                self.rows.append("edt_2")
                self.rows.append("edt_indicator_2")
                self.rows.append("sup")
                self.rows.append("pred")
                self.rows.append("pred_consume")
            self.rows.append("pred_consume_invariant")
            self.rows.append("pred_consume_invariant_beta_unifroms15")
            self.rows.append("pred_consume_invariant_beta_rnt_unifroms15")
            if is_baseline:
                self.rows.append("end2end")

            batch_metas.append(metas)
            batch_commons.append(commons)
            batch_metrics.append(metrics)
            batch_waypoints.append(waypoints)
            batch_trajs.append(trajs)

            if DEMO_VIS or DEMO_SAVE:
                figure = plt.figure(figsize=(30, 15))
                color_tree = np.array([0.,0.,0.])
                demo_scan = self.visulizater.naive_vis(bev_map)
                demo_segmentation = self.visulizater.stack_layer(self.visulizater.defined_label_vis(bev_label), 
                    [binary_traversable], 
                    fg_color_list=[np.array([1.,1.,1.])])
                demo_registration = self.visulizater.stack_layer(demo_segmentation, 
                    [eline_points_map], 
                    fg_color_list=[np.array([1.,0.,0.])])
                demo_planner_ogm = self.visulizater.stack_layer(demo_registration, 
                    (rrt_tree_ogm > 0,self.convertor.matrix_point2map(greedy_points_ogm[0], radius=4),self.convertor.matrix_point2map(waypoint_ogm, radius=4))+self.convertor.matrix_list2map(rrt_traj_ogm, edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([0.,1.,1.]),np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1])
                color_tree = np.array([0.,0.,0.,0.5])
                demo_planner_pred_beta = self.visulizater.tangent_vis(tangent_map_pred_beta, pltcm='hsv')
                axesimg = figure.add_subplot(2, 4, 1)
                axesimg.axis("off")
                axesimg.imshow(demo_scan)
                axesimg = figure.add_subplot(2, 4, 2)
                axesimg.axis("off")
                axesimg.imshow(demo_segmentation)
                axesimg = figure.add_subplot(2, 4, 3)
                axesimg.axis("off")
                axesimg.imshow(demo_registration)
                axesimg = figure.add_subplot(2, 4, 4)
                axesimg.axis("off")
                axesimg.imshow(demo_planner_ogm)
                axesimg = figure.add_subplot(2, 4, 5)
                axesimg.axis("off")
                axesimg.imshow(demo_planner_pred_beta)
                axesimg = figure.add_subplot(2, 4, 6)
                axesimg.axis("off")
                axesimg.imshow(demo_planner_pred_beta_2)
                axesimg = figure.add_subplot(2, 4, 7)
                axesimg.axis("off")
                axesimg.imshow(demo_planner_pred_beta_3)

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
                if DEMO_VIS:
                    plt.show()
                if DEMO_SAVE:
                    plt.savefig(self.get_save_dir("bev", "demo", seq_id, frame_id, "png"), transparent=True)
                    plt.close()

            if DEPLOY_SAVE or DEBUG_VIS or FAILURE_SAVE and failure or FAILURE_VIS and failure:
                figure = plt.figure(figsize=(30, 15))
                color_tree = np.array([0/255,128/255,255/255])
                interp_target_points_map = self.convertor.matrix_list2map(self.convertor.car2matrix(target_interp))
                colored_bg = self.visulizater.stack_layer(self.visulizater.defined_label_vis(bev_label), 
                    [binary_traversable, self.convertor.matrix_point2map(waypoint_target, radius=4), interp_target_points_map, eline_points_map], 
                    fg_color_list=[np.array([1.,1.,1.]), np.array([0.,0.,1.]), np.array([0.,0.,0.]), np.array([1.,0.,0.])])
                full_colored_bg = self.visulizater.stack_layer(self.visulizater.defined_label_vis(full_bev_label), 
                    [full_binary_traversable, self.convertor.matrix_point2map(waypoint_target, radius=4), interp_target_points_map, eline_points_map], 
                    fg_color_list=[np.array([1.,1.,1.]), np.array([0.,0.,1.]), np.array([0.,0.,0.]), np.array([1.,0.,0.])])

                axesimg = figure.add_subplot(5, 10, 1)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(self.visulizater.naive_vis(bev_map), [eline_points_map, curve_points_map], fg_color_list=[np.array([1.,0.,0.]), np.array([1.,0.,0.])]))
                axesimg = figure.add_subplot(5, 10, 11)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (self.convertor.matrix_point2map(greedy_points_straight[0], radius=4),self.convertor.matrix_point2map(waypoint_straight, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_straight), edge=True)[::-1], 
                    fg_color_list=(np.array([0.,1.,1.]),np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 21)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (skeleton_binary_traversable,self.convertor.matrix_point2map(waypoint_skeleton, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_skeleton), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 2)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_ogm > 0,self.convertor.matrix_point2map(greedy_points_ogm[0], radius=4),self.convertor.matrix_point2map(waypoint_ogm, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_ogm), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([0.,1.,1.]),np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 12)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_ogm_frontier > 0,self.convertor.matrix_list2map(greedy_points_ogm_frontier),self.convertor.matrix_point2map(waypoint_ogm_frontier, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_ogm_frontier), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([0.,1.,1.]),np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 22)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_ogm_skeleton_frontier > 0,self.convertor.matrix_list2map(greedy_points_ogm_skeleton_frontier),self.convertor.matrix_point2map(waypoint_ogm_skeleton_frontier, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_ogm_skeleton_frontier), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([0.,1.,1.]),np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))

                axesimg = figure.add_subplot(5, 10, 3)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_simple, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 4)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_naive, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 23)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_baseline, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 33)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_baseline_1, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 35)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_edt_1, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 37)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_edt_indicator_1, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 43)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_baseline_2, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 45)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_edt_2, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 47)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_edt_indicator_2, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 13)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_simple > 0,self.convertor.matrix_point2map(waypoint_simple, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_simple), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 14)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_naive > 0,self.convertor.matrix_point2map(waypoint_naive, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_naive), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 24)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_baseline > 0,self.convertor.matrix_point2map(waypoint_baseline, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_baseline), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 34)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_baseline_1 > 0,self.convertor.matrix_point2map(waypoint_baseline_1, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_baseline_1), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 36)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_edt_1 > 0,self.convertor.matrix_point2map(waypoint_edt_1, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_edt_1), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 38)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_edt_indicator_1 > 0,self.convertor.matrix_point2map(waypoint_edt_indicator_1, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_edt_indicator_1), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 44)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(full_colored_bg, 
                    (rrt_tree_baseline_2 > 0,self.convertor.matrix_point2map(waypoint_baseline_2, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_baseline_2), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 46)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(full_colored_bg, 
                    (rrt_tree_edt_2 > 0,self.convertor.matrix_point2map(waypoint_edt_2, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_edt_2), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 48)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(full_colored_bg, 
                    (rrt_tree_edt_indicator_2 > 0,self.convertor.matrix_point2map(waypoint_edt_indicator_2, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_edt_indicator_2), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))

                axesimg = figure.add_subplot(5, 10, 5)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_init, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 6)
                axesimg.axis("off")
                axesimg.imshow(distance_init)

                axesimg = figure.add_subplot(5, 10, 16)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_pred > 0,self.convertor.matrix_point2map(waypoint_pred, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_pred), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 25)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_pred_consume > 0,self.convertor.matrix_point2map(waypoint_pred_consume, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_pred_consume), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))

                axesimg = figure.add_subplot(5, 10, 15)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_pred, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 17)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_pred_beta, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 19)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_pred_beta_rnt, pltcm='hsv'))
                axesimg = figure.add_subplot(5, 10, 20)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.tangent_vis(tangent_map_pred_alpha, pltcm='hsv'))

                axesimg = figure.add_subplot(5, 10, 26)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (rrt_tree_pred_consume_invariant > 0,self.convertor.matrix_point2map(waypoint_pred_consume_invariant, radius=4))+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_pred_consume_invariant), edge=True)[::-1], 
                    fg_color_list=(color_tree,np.array([1.,0.,0.]))+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))
                axesimg = figure.add_subplot(5, 10, 27)
                axesimg.axis("off")
                axesimg = figure.add_subplot(5, 10, 29)
                axesimg.axis("off")
                axesimg = figure.add_subplot(5, 10, 10)
                axesimg.axis("off")
                axesimg.imshow(self.visulizater.stack_layer(colored_bg, 
                    (self.convertor.matrix_point2map(waypoint_end2end, radius=4),)+self.convertor.matrix_list2map(self.convertor.car2matrix(pred_interp_end2end), edge=True)[::-1], 
                    fg_color_list=(np.array([1.,0.,0.]),)+(np.array([1.,0.,0.]), np.array([0.,1.,0.]))[::-1]))

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

                if DEBUG_VIS or FAILURE_VIS and failure:
                    plt.show()
                if DEPLOY_SAVE:
                    plt.savefig(self.get_save_dir("bev", "combine.interpolate", seq_id, frame_id, "png"))
                    plt.close()
                    np.save(self.get_save_dir("bev", "tangent_map_pred_beta", seq_id, frame_id, "npy"), tangent_map_pred_beta)
                    save_tensor_as_image(self.visulizater.tangent_vis(tangent_map_pred_beta, pltcm='hsv')[..., :3], self.get_save_dir("bev", "tangent_map_pred_beta", seq_id, frame_id, "png"))
                if FAILURE_SAVE and failure:
                    plt.savefig(self.get_save_dir("bev", "failure.interpolate", seq_id, frame_id, "png"))
                    plt.close()

            timer.get()

            timer.flush()
        return batch_metas, batch_commons, batch_metrics, batch_waypoints, batch_trajs

    def dictlist_to_listdict(self, dictlist):
        listdict = {}
        for dict in dictlist:
            for k, v in dict.items():
                if k not in listdict.keys():
                    listdict[k] = []
                listdict[k].append(v)
        return listdict

    def evaluate(self):
        # self.pool.close()  # No more tasks will be submitted to the pool
        # self.pool.join()   # Wait for the worker processes to finish

        if self._distributed:
            synchronize()
            self._metas = list(itertools.chain(*all_gather(self._metas)))
            self._commons = list(itertools.chain(*all_gather(self._commons)))
            self._metrics = list(itertools.chain(*all_gather(self._metrics)))
            self._waypoints = list(itertools.chain(*all_gather(self._waypoints)))
            self._trajs = list(itertools.chain(*all_gather(self._trajs)))

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

        ordered_dict = {}

        metas = self.dictlist_to_listdict(self._metas)
        seq_ids = metas["seq_id"]
        frame_ids = metas["frame_id"]
        poses = metas["pose"]

        seq_meta = {}
        for i in range(len(seq_ids)):
            seq_id = seq_ids[i]
            meta = self._metas[i]
            if seq_id not in seq_meta.keys():
                seq_meta[seq_id] = []
            seq_meta[seq_id].append({
                "meta": meta,
            })
        if self._output_dir:
            for seq_id, seq_metas in seq_meta.items():
                metas_dict = self.dictlist_to_listdict(seq_metas)
                seq_pose = self.dictlist_to_listdict(metas_dict["meta"])["pose"]
                file_path = os.path.join(self._output_dir, f"pose_{seq_id}.txt")
                with PathManager.open(file_path, "w") as f:
                    f.write('\n'.join([' '.join(map(str, item[:3].reshape(-1).tolist())) for item in seq_pose]))

        seq_metric = {}
        res_seq_metric = {}
        for i in range(len(seq_ids)):
            seq_id = seq_ids[i]
            common = self._commons[i]
            metric = self._metrics[i]
            if seq_id not in seq_metric.keys():
                seq_metric[seq_id] = []
            if "all" not in seq_metric.keys():
                seq_metric["all"] = []
            seq_metric[seq_id].append({
                "common": common,
                "metric": metric,
            })
            seq_metric["all"].append({
                "common": common,
                "metric": metric,
            })
        for seq_id, seq_commons_metrics in seq_metric.items():
            res_seq_metric[seq_id] = {}
            commons_metric_dict = self.dictlist_to_listdict(seq_commons_metrics)
            for method, commons in self.dictlist_to_listdict(commons_metric_dict["common"]).items():
                for common, results in self.dictlist_to_listdict(commons).items():
                    res_seq_metric[seq_id][f"{common}_{method}"] = np.array(results).mean()
            for method, metrics in self.dictlist_to_listdict(commons_metric_dict["metric"]).items():
                for metric, results in self.dictlist_to_listdict(metrics).items():
                    res_seq_metric[seq_id][f"{metric}_{method}"] = np.array(results).mean()
        if self._output_dir:
            for seq_id, res in res_seq_metric.items():
                print("seq_id, res", seq_id, res)
                file_path = os.path.join(self._output_dir, f"trajectory_evaluation_{seq_id}.txt")
                with PathManager.open(file_path, "w") as f:
                    metrics = np.zeros((len(self.rows), len(self.cols)))
                    for i, a in enumerate(self.rows):
                        for j, b in enumerate(self.cols):
                            metrics[i, j] = res[b+"_"+a]
                    metrics_txt = ","+",".join(self.cols)+"\n"
                    f.write(metrics_txt+"\n".join([self.rows[i]+","+ ",".join(map(str, m.tolist())) for i, m in enumerate(metrics)]))

        seq_useful = {}
        for i in range(len(seq_ids)):
            seq_id = seq_ids[i]
            waypoint = self._waypoints[i]
            traj = self._trajs[i]
            if seq_id not in seq_useful.keys():
                seq_useful[seq_id] = []
            seq_useful[seq_id].append({
                "waypoint": waypoint,
                "traj": traj,
            })
        if self._output_dir:
            for seq_id, seq_usefuls in seq_useful.items():
                seq_path = os.path.join(self._output_dir, seq_id)
                if not os.path.exists(seq_path):
                    os.makedirs(seq_path)
                usefuls_dict = self.dictlist_to_listdict(seq_usefuls)
                for method, waypoints in self.dictlist_to_listdict(usefuls_dict["waypoint"]).items():
                    for waypoint, results in self.dictlist_to_listdict(waypoints).items():
                        wp_pd_tg = np.array(results)
                        file_path = os.path.join(seq_path, f"wp_pd_tg_-{method}-{waypoint}.txt")
                        with PathManager.open(file_path, "w") as f:
                            f.write('\n'.join([' '.join(map(str, item.tolist())) for item in wp_pd_tg]))
                for method, trajs in self.dictlist_to_listdict(usefuls_dict["traj"]).items():
                    for traj, results in self.dictlist_to_listdict(trajs).items():
                        tj_pd_tg = results
                        file_path = os.path.join(seq_path, f"tj_pd_tg_-{method}-{traj}.txt")
                        with PathManager.open(file_path, "w") as f:
                            f.write('\n'.join([' '.join(map(str, item.reshape(-1).tolist())) for item in tj_pd_tg]))
            for seq_id, seq_usefuls in seq_useful.items():
                try:
                    guidance_path = os.path.join(os.getenv('DATASET_PATH'), seq_id, "guidance_osm.txt")
                    map_path = os.path.join(os.getenv('DATASET_PATH'), seq_id, "terrainMapPGO.ply")
                    instruction_path = os.path.join(os.getenv('DATASET_PATH'), seq_id, "instructions.txt")
                    plot_waypoint(self._output_dir, seq_id, guidance_path=guidance_path, map_path=None)
                    plot_histogram(self._output_dir, seq_id, guidance_path=guidance_path, instruction_path=instruction_path, end_to_end_path=None)
                except Exception as e:
                    print("Exception from plot:", e)

        ordered_dict["trajectory"] = res_seq_metric["all"]
        results = OrderedDict(ordered_dict)
        self._logger.info(results)
        return results
