# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import logging

import torch

from detectron2.config import configurable
from detectron2.data import DatasetCatalog, MetadataCatalog

from dataloader.dataset_loader import trainset_loader, valset_loader
from common.polar import *
from common.peroidic import peroidic_np
from common.envs import *


def get_dataset_dicts(train):
    if train:
        loader = trainset_loader
    else:
        loader = valset_loader

    dataset_dicts = []
    for index in range(DEBUG_START, len(loader)):
        record = {}
        record["id"] = index
        record["width"] = -1
        record["height"] = -1
        filename = loader.dataset.get_frame_name(index)
        record["file_name"] = filename
        record["sem_seg_file_name"] = filename
        record["pan_seg_file_name"] = filename
        seq_id, frame_id = loader.dataset.parsePathInfoByIndex(index)
        record["seq_id"], record["frame_id"] = seq_id, frame_id

        record["res"] = loader.res
        record["lidar_range"] = loader.lidar_range

        dataset_dicts.append(record)
    return dataset_dicts


def set_meta(train_name, val_name, trainset, valset):
    MetadataCatalog.get(train_name).set(cls_weight=trainset.cls_weight)
    MetadataCatalog.get(train_name).set(cls_alpha=trainset.cls_alpha)
    MetadataCatalog.get(train_name).set(config=trainset.data_config)
    MetadataCatalog.get(train_name).set(class_map_lut_inv=trainset.class_map_lut_inv)
    MetadataCatalog.get(train_name).set(color_map=trainset.sem_color_lut)
    MetadataCatalog.get(val_name).set(cls_weight=valset.cls_weight)
    MetadataCatalog.get(val_name).set(cls_alpha=valset.cls_alpha)
    MetadataCatalog.get(val_name).set(config=valset.data_config)
    MetadataCatalog.get(val_name).set(class_map_lut_inv=valset.class_map_lut_inv)
    MetadataCatalog.get(val_name).set(color_map=valset.sem_color_lut)

    MetadataCatalog.get(train_name).set(thing_dataset_id_to_contiguous_id=trainset.thing_dataset_id_to_contiguous_id)
    MetadataCatalog.get(train_name).set(stuff_dataset_id_to_contiguous_id=trainset.stuff_dataset_id_to_contiguous_id)
    MetadataCatalog.get(train_name).set(stuff_classes=trainset.stuff_classes)
    MetadataCatalog.get(train_name).set(ignore_label=trainset.ignore_label)
    MetadataCatalog.get(train_name).set(evaluator_type=trainset.evaluator_type)
    MetadataCatalog.get(val_name).set(thing_dataset_id_to_contiguous_id=valset.thing_dataset_id_to_contiguous_id)
    MetadataCatalog.get(val_name).set(stuff_dataset_id_to_contiguous_id=valset.stuff_dataset_id_to_contiguous_id)
    MetadataCatalog.get(val_name).set(stuff_classes=valset.stuff_classes)
    MetadataCatalog.get(val_name).set(ignore_label=valset.ignore_label)
    MetadataCatalog.get(val_name).set(evaluator_type=valset.evaluator_type)


set_meta("trainset", "valset", trainset_loader.dataset, valset_loader.dataset)
DatasetCatalog.register("trainset", lambda t=True: get_dataset_dicts(t))
DatasetCatalog.register("valset", lambda t=False: get_dataset_dicts(t))


__all__ = ["DatasetMapper"]


class DatasetMapper:
    @configurable
    def __init__(
        self,
        is_train=True,
    ):
        self.is_train = is_train

        if self.is_train:
            self.loader = trainset_loader
        else:
            self.loader = valset_loader

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        augmentations = None
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        ret = {
            "is_train": is_train,
        }
        return ret

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        t = self.loader[dataset_dict["id"]]
        dataset_dict["bev_map"] = torch.tensor(t["bev_map"], dtype=torch.float32)
        dataset_dict["tangent_init"] = torch.tensor(t["tangent_init"], dtype=torch.float32)
        dataset_dict["distance_init"] = torch.tensor(t["distance_init"], dtype=torch.float32)
        dataset_dict["tangent_raw"] = torch.tensor(t["tangent_raw"], dtype=torch.float32)
        dataset_dict["tangent_map"] = torch.tensor(t["tangent_map"], dtype=torch.float32)
        dataset_dict["distance_map"] = torch.tensor(t["distance_map"], dtype=torch.float32)
        dataset_dict["training_mask"] = torch.tensor(t["training_mask"], dtype=torch.bool)

        dataset_dict["bev_label"] = torch.tensor(t["bev_label"], dtype=torch.int64)
        dataset_dict["label_mask"] = torch.tensor(t["label_mask"], dtype=torch.bool)
        dataset_dict["label_padded"] = torch.tensor(t["label_padded"], dtype=torch.int64)
        dataset_dict["binary_traversable"] = torch.tensor(t["binary_traversable"], dtype=torch.bool)
        dataset_dict["full_bev_label"] = torch.tensor(t["full_bev_label"], dtype=torch.int64)
        dataset_dict["full_label_mask"] = torch.tensor(t["full_label_mask"], dtype=torch.bool)
        dataset_dict["full_label_padded"] = torch.tensor(t["full_label_padded"], dtype=torch.int64)
        dataset_dict["full_binary_traversable"] = torch.tensor(t["full_binary_traversable"], dtype=torch.bool)

        dataset_dict["eline_points"] = torch.tensor(t["eline_points"], dtype=torch.int32)
        dataset_dict["eline_points_map"] = torch.tensor(t["eline_points_map"], dtype=torch.bool)
        dataset_dict["eline_tangents"] = torch.tensor(t["eline_tangents"], dtype=torch.float32)
        dataset_dict["eline_tangents_map"] = torch.tensor(t["eline_tangents_map"], dtype=torch.float32)
        dataset_dict["curve_points"] = torch.tensor(t["curve_points"], dtype=torch.int32)
        dataset_dict["curve_points_map"] = torch.tensor(t["curve_points_map"], dtype=torch.bool)
        dataset_dict["curve_tangents"] = torch.tensor(t["curve_tangents"], dtype=torch.float32)
        dataset_dict["curve_tangents_map"] = torch.tensor(t["curve_tangents_map"], dtype=torch.float32)

        dataset_dict["greedy_point"] = torch.tensor(t["greedy_point"], dtype=torch.int64)
        dataset_dict["greedy_points"] = torch.tensor(t["greedy_points"], dtype=torch.int64)
        dataset_dict["greedy_tangents"] = torch.tensor(t["greedy_tangents"], dtype=torch.float32)

        dataset_dict["target_points"] = torch.tensor(t["target_points"], dtype=torch.int32)
        dataset_dict["target_traj"] = torch.tensor(t["target_traj"], dtype=torch.float32)
        dataset_dict["target_points_map"] = torch.tensor(t["target_points_map"], dtype=torch.bool)
        dataset_dict["target_points_past"] = torch.tensor(t["target_points_past"], dtype=torch.int32)
        dataset_dict["target_traj_past"] = torch.tensor(t["target_traj_past"], dtype=torch.float32)
        dataset_dict["target_points_past_map"] = torch.tensor(t["target_points_past_map"], dtype=torch.bool)

        tangent_offset = polar_sub(t["tangent_map"], t["tangent_raw"])  # -2pi, 2pi
        tangent_offset = peroidic_np(tangent_offset, norm=True, abs=False)  # -1, 1
        tangent_offset[~t["training_mask"]] = 0
        dataset_dict["regression"] = torch.tensor(tangent_offset, dtype=torch.float32)
        dataset_dict["regression2"] = dataset_dict["regression"][..., None].expand(dataset_dict["regression"].shape + (2,))
        tangent_reconstruction = polar_sub(t["tangent_map"], t["tangent_init"])  # -2pi, 2pi
        tangent_reconstruction = peroidic_np(tangent_reconstruction, norm=True, abs=False)  # -1, 1
        tangent_reconstruction[~t["training_mask"]] = 0
        dataset_dict["regression3"] = torch.tensor(tangent_reconstruction, dtype=torch.float32)
        distance_map = t["distance_map"]
        distance_map[~t["training_mask"]] = 0
        dataset_dict["regression4"] = torch.tensor(distance_map, dtype=torch.float32)

        if "pose" in t.keys():
            dataset_dict["pointcloud"] = t["pointcloud"]
            dataset_dict["sem_label"] = t["sem_label"]
            dataset_dict["pose"] = t["pose"]

        return dataset_dict
