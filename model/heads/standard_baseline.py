# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

# from .pixel_decoder import build_pixel_decoder
from .boundary_loss import BoundaryLoss

from api.datasetapi.dataset.dataset import pc_processor


@SEM_SEG_HEADS_REGISTRY.register()
class StandardBaselineHead(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            logger = logging.getLogger(__name__)
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "sem_seg_head" in k and not k.startswith(prefix + "predictor"):
                    newk = k.replace(prefix, prefix + "pixel_decoder.")
                    # logger.warning(f"{k} ==> {newk}")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        pixel_decoder: nn.Module,
        cls_weight,
        cls_alpha,
        num_classes: int,
        ignore_value: int = -1,
        num_deep_levels,
        direct_pred,
        no_focal,
        class_balance,
        loss_weight: float = 1.0,
        lovasz,
        lovasz_weight,
        boundary,
        boundary_weight,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
        """
        super().__init__()
        if not direct_pred:
            input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
            self.in_features = [k for k, v in input_shape]
            feature_strides = [v.stride for k, v in input_shape]
            feature_channels = [v.channels for k, v in input_shape]
            self.common_stride = 4
        if not direct_pred:
            predictor_in_channels = pixel_decoder.mask_dim
        else:
            predictor_in_channels = input_shape[1]
        self.pixel_decoder = pixel_decoder
        self.register_buffer("cls_weight", torch.Tensor(cls_weight), False)
        self.register_buffer("cls_alpha", torch.Tensor(cls_alpha), False)

        self.ignore_value = ignore_value

        self.num_deep_levels = num_deep_levels
        if num_deep_levels == 1:
            predictor = Conv2d(
                predictor_in_channels, num_classes, kernel_size=1, stride=1, padding=0
            )
        elif num_deep_levels > 1:
            predictor = nn.ModuleList()
            for i in range(num_deep_levels):
                predictor.append(Conv2d(
                    predictor_in_channels, num_classes, kernel_size=1, stride=1, padding=0
                ))
        else:
            predictor = None
        if num_deep_levels == 1:
            self.predictor = predictor
            weight_init.c2_msra_fill(self.predictor)
        elif num_deep_levels > 1:
            self.predictor = predictor
            for i in range(num_deep_levels):
                weight_init.c2_msra_fill(self.predictor[i])
        else:
            self.predictor = None

        self.direct_pred = direct_pred
        self.no_focal = no_focal
        self.class_balance = class_balance
        self.loss_weight = loss_weight
        self.lovasz = lovasz
        self.lovasz_weight = lovasz_weight
        self.boundary = boundary
        self.boundary_weight = boundary_weight

        if not self.no_focal:
            if not self.class_balance:
                self.sig_focal_loss = pc_processor.loss.FocalSoftmaxLoss(num_classes, gamma=2, softmax=False)
            else:
                self.sig_focal_loss = pc_processor.loss.FocalSoftmaxLoss(num_classes, gamma=2, alpha=cls_alpha, softmax=False)
        self.lovasz_loss = pc_processor.loss.Lovasz_softmax(ignore=ignore_value)
        self.boundary_loss = BoundaryLoss()

    @classmethod
    def from_config(cls, cfg, input_shape):
        direct_pred = cfg.MODEL.SEM_SEG_HEAD.DIRECT_PRED
        if not direct_pred:
            input_shape_dict = {
                k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
            }
            pixel_decoder = build_pixel_decoder(cfg, input_shape)
        else:
            input_shape_dict = input_shape
            pixel_decoder = None
        cls_weight = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("cls_weight")
        print("len, cls_weight", len(cls_weight), cls_weight)
        cls_alpha = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]).get("cls_alpha")
        print("len, cls_alpha", len(cls_alpha), cls_alpha)

        return {
            "input_shape": input_shape_dict,
            "pixel_decoder": pixel_decoder,
            "cls_weight": cls_weight,
            "cls_alpha": cls_alpha,

            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,

            "num_deep_levels": cfg.MODEL.SEM_SEG_HEAD.NUM_DEEP_LEVELS,

            "direct_pred": direct_pred,
            "no_focal": cfg.MODEL.SEM_SEG_HEAD.NO_FOCAL,
            "class_balance": cfg.MODEL.SEM_SEG_HEAD.CLASS_BALANCE,
            "loss_weight": cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT,
            "lovasz": cfg.MODEL.SEM_SEG_HEAD.LOVASZ,
            "lovasz_weight": cfg.MODEL.SEM_SEG_HEAD.LOVASZ_WEIGHT,
            "boundary": cfg.MODEL.SEM_SEG_HEAD.BOUNDARY,
            "boundary_weight": cfg.MODEL.SEM_SEG_HEAD.BOUNDARY_WEIGHT,
        }

    def forward(self, features, mask_features, targets=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        x = self.layers(features, mask_features)
        if self.training:
            if not self.direct_pred:
                upx = F.interpolate(
                    x[-1], scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
                return upx, self.losses(x, targets)
            else:
                upx = F.interpolate(
                    x[-1], size=targets.shape[-2:], mode="bilinear", align_corners=False
                )
                return upx, self.losses(x, targets)
        else:
            if not self.direct_pred:
                upx = F.interpolate(
                    x[-1], scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
            else:
                upx = F.interpolate(
                    x[-1], size=targets.shape[-2:], mode="bilinear", align_corners=False
                )
            return upx, {}

    def layers(self, features, mask_features):
        if not self.direct_pred:
            x, _ = self.pixel_decoder.forward_features(features)
        else:
            x = mask_features
        if not isinstance(x, list):
            x = [self.predictor(x)]
        else:
            for i in range(len(x)):
                x[i] = self.predictor[i](x[i])
        return x

    def losses(self, predictions_list, targets):
        losses = {}
        for i, predictions in enumerate(predictions_list):
            predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
            if not self.direct_pred:
                predictions = F.interpolate(
                    predictions, scale_factor=self.common_stride, mode="bilinear", align_corners=False
                )
            else:
                predictions = F.interpolate(
                    predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False
                )
            if not self.no_focal:
                loss = self.sig_focal_loss(predictions.softmax(dim=1), targets, mask=targets != self.ignore_value)
            else:
                if not self.class_balance:
                    loss = F.cross_entropy(
                        predictions, targets, reduction="mean", ignore_index=self.ignore_value
                    )
                else:
                    loss = F.cross_entropy(
                        predictions, targets, reduction="mean", ignore_index=self.ignore_value, weight=self.cls_weight
                    )
            if not self.lovasz:
                loss_lovasz = torch.tensor(0.0)
            else:
                loss_lovasz = self.lovasz_loss(predictions.softmax(dim=1), targets)
            if not self.boundary:
                loss_boundary = torch.tensor(0.0)
            else:
                loss_boundary = self.boundary_loss(predictions.softmax(dim=1), targets)
            l_dict = {"loss_sem_seg": loss * self.loss_weight, "loss_lovasz": loss_lovasz * self.lovasz_weight, "loss_boundary": loss_boundary * self.boundary_weight}
            if i != len(predictions_list) -1:
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses
