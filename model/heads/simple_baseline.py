# Copyright (c) Facebook, Inc. and its affiliates.
import logging

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from common.peroidic import peroidic_torch
from .boundary_loss import BoundaryLoss

from api.datasetapi.dataset.dataset import pc_processor


class TangentEstimation(nn.Module):
    def __init__(self, base_channels):
        super(TangentEstimation, self).__init__()
        self.conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_channels, 1, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.InstanceNorm2d(base_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class ShiftEstimation(nn.Module):
    def __init__(self, base_channels):
        super(ShiftEstimation, self).__init__()
        self.conv1 = nn.Conv2d(base_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(base_channels, 2, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.InstanceNorm2d(base_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


@SEM_SEG_HEADS_REGISTRY.register()
class SimpleBaselineHead(nn.Module):

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
        # cls_weight,
        # cls_alpha,
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
        predictor_in_channels = input_shape[1]
        self.pixel_decoder = pixel_decoder

        self.ignore_value = ignore_value

        self.num_deep_levels = num_deep_levels
        predictor_c = 1
        predictor2_c = 2
        if num_deep_levels == 1:
            predictor = Conv2d(
                predictor_in_channels, num_classes, kernel_size=1, stride=1, padding=0
            )
            regressor = Conv2d(
                predictor_in_channels, predictor_c, kernel_size=1, stride=1, padding=0
            )
            regressor2 = Conv2d(
                predictor_in_channels, predictor2_c, kernel_size=1, stride=1, padding=0
            )
        elif num_deep_levels > 1:
            predictor = nn.ModuleList()
            regressor = nn.ModuleList()
            regressor2 = nn.ModuleList()
            for i in range(num_deep_levels):
                predictor.append(Conv2d(
                    predictor_in_channels, num_classes, kernel_size=1, stride=1, padding=0
                ))
                regressor.append(Conv2d(
                    predictor_in_channels, predictor_c, kernel_size=1, stride=1, padding=0
                ))
                regressor2.append(Conv2d(
                    predictor_in_channels, predictor2_c, kernel_size=1, stride=1, padding=0
                ))
        else:
            predictor = None
            regressor = None
            regressor2 = None
        self.regressor = regressor

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

        input_shape_dict = input_shape
        pixel_decoder = None

        return {
            "input_shape": input_shape_dict,
            "pixel_decoder": pixel_decoder,

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

    def forward(self, features, mask_features, targets=None, targets_regressor=None, targets_regressor_mask=None, tangent_init_polar=None):
        """
        Returns:
            In training, returns (None, dict of losses)
            In inference, returns (CxHxW logits, {})
        """
        y = self.layers_regressor(features, mask_features)
        if self.training:
            upy = F.interpolate(
                y[-1], size=targets_regressor.shape[-2:], mode="bilinear", align_corners=False
            )
            return upy, self.losses_regressor(y, targets_regressor, targets_regressor_mask, tangent_init_polar)
        else:
            # return x[-1], {}
            return y[-1], {}

    def layers(self, features, mask_features):
        x = mask_features
        if not isinstance(x, list):
            x = [self.predictor(x)]
        else:
            for i in range(len(x)):
                x[i] = self.predictor[i](x[i])
        return x

    def layers_regressor(self, features, mask_features):
        x = mask_features
        if not isinstance(x, list):
            x = [self.regressor(x)]
        else:
            for i in range(len(x)):
                x[i] = self.regressor[i](x[i])
        return x

    def losses(self, predictions_list, targets):
        losses = {}
        for i, predictions in enumerate(predictions_list):
            predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
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

    def decompose_predictions(self, predictions):
        tangent_res_pred_polar = predictions[:, 0]
        shift_res_pred = predictions[:, 1:3].permute(0, 2, 3, 1)
        return tangent_res_pred_polar, shift_res_pred
        
    def losses_regressor(self, predictions_list, targets, targets_mask, tangent_init_polar):
        losses = {}
        for i, predictions in enumerate(predictions_list):
            predictions = predictions.float()  # https://github.com/pytorch/pytorch/issues/48163
            # print("predictions.shape, targets.shape", predictions.shape, targets.shape)
            predictions = F.interpolate(
                predictions, size=targets.shape[-2:], mode="bilinear", align_corners=False
            )

            diff = predictions - targets[:, 3:4]
            diff = peroidic_torch(diff, norm=False, abs=True).mean(1)
            loss = diff[targets_mask].sum() / targets_mask.sum() if targets_mask.sum() > 0 else diff[targets_mask].sum()

            l_dict = {"loss_regression": loss}
            if i != len(predictions_list) -1:
                l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
            losses.update(l_dict)
        return losses
