import os
import numpy as np

from .utils import get_panoptic_weight_alpha, get_w1_w2_long_tail_sem

from .dataset import waymo_open_dataset_loader


class Waymo(waymo_open_dataset_loader.dataset.Waymo):
    def __init__(self, 
                 panoptic, 
                 root,
                 version='v2.0.0',
                 split='train',
                 has_image=True,
                 has_pcd=True,
                 has_label=True):
        super(Waymo, self).__init__(root=root, version=version, split=split, has_image=has_image, has_label=has_label)

        print("[panoptic: {}]".format(panoptic))

        # addition
        self.data_config = None

        # meta
        stuff_classes = ["Undefined",
                            "Car", "Truck", "Bus", "Other Vehicle", "Motorcyclist", "Bicyclist", "Pedestrian",
                            "Sign", "Traffic Light", "Pole", "Construction Cone", 
                            "Bicycle", "Motorcycle", 
                            "Building", "Vegetation", "Tree Trunk", "Curb", "Road", "Lane Marker", "Other Ground", "Walkable", "Sidewalk"]
        thing_dataset_id_to_contiguous_id = {k: k for i, k in enumerate([1, 2, 3, 4, 5, 6, 7, 12, 13])}
        stuff_dataset_id_to_contiguous_id = {k: k for i, k in enumerate(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])}
        self.thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
        self.stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
        self.stuff_classes = stuff_classes
        self.ignore_label = 0
        if not panoptic:
            self.evaluator_type = "sem_seg"
        else:
            self.evaluator_type = "ade20k_panoptic_seg"

        # weight
        reweight = [1]*23

        cls_weight = 1 / (self.cls_freq + 1e-3)
        cls_weight[0] = 0
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
