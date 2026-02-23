import os
import numpy as np

from .utils import get_panoptic_weight_alpha, get_w1_w2_long_tail_sem

from .dataset import um_osmplanner


class UMOSMPlanner(um_osmplanner.UMOSMPlannerWithPose):
    def __init__(self, 
                 panoptic, 
                 root,
                 splits,
                 mode='terrain'):
        super(UMOSMPlanner, self).__init__(root=root, splits=splits, mode=mode)

        print("[panoptic: {}]".format(panoptic))

        # addition
        self.data_config = ""

        # meta
        stuff_classes = ["fence",
                            "vegetation", "road"]
        thing_dataset_id_to_contiguous_id = {k: k for i, k in enumerate([])}
        stuff_dataset_id_to_contiguous_id = {k: k for i, k in enumerate(
            [0, 1, 2, 3])}
        self.thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
        self.stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
        self.stuff_classes = stuff_classes
        self.ignore_label = 0
        if not panoptic:
            self.evaluator_type = "sem_seg"
        else:
            self.evaluator_type = "ade20k_panoptic_seg"

        # weight
        reweight = [1, 1, 1, 1]

        cls_weight = 1 / (self.cls_freq + 1e-3)
        for cl, w in enumerate(cls_weight):
            if cl == self.ignore_label:
                cls_weight[cl] = 0
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
