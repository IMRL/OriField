import numpy as np


def get_panoptic_weight_alpha(cls_weight, reweight):
    cls_weight_panoptic = cls_weight * reweight
    cls_alpha_panoptic = np.log(1 + cls_weight_panoptic)
    cls_alpha_panoptic = cls_alpha_panoptic / cls_alpha_panoptic.max()
    return cls_weight_panoptic, cls_alpha_panoptic


def get_w1_w2_long_tail_sem(panoptic, long_tail, cls_weight, cls_alpha, cls_weight_panoptic, cls_alpha_panoptic):
    if not panoptic:
        w1 = cls_weight / cls_weight.max()
        w2 = cls_alpha
        long_tailed_sem = (cls_weight / cls_weight.max() > long_tail).nonzero()[0]
    else:
        w1 = cls_weight_panoptic / cls_weight_panoptic.max()
        w2 = cls_alpha_panoptic
        long_tailed_sem = (cls_weight_panoptic / cls_weight_panoptic.max() > long_tail).nonzero()[0]
    return w1, w2, long_tailed_sem
