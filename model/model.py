# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone

from model.heads import SimpleBaselineHead

from common.polar import cartesian_to_polar
from common.peroidic import peroidic_torch


@META_ARCH_REGISTRY.register()
class Model(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head

        self.register_buffer("pixel_mean", torch.Tensor([0, 0, 0]).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor([1, 1, 1]).view(-1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_directer(cfg)
        sem_seg_head = build_sem_seg_head(cfg, (backbone.bottleneck_channels, backbone.base_channels))
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        tangent_init_polar = torch.stack([cartesian_to_polar(input["tangent_init"][..., 0], input["tangent_init"][..., 1])[1] for input in batched_inputs]).to(self.device)  # -pi, pi
        tangent_init_polar = peroidic_torch(tangent_init_polar, norm=True, abs=False)  # -1, 1
        bev_map = torch.stack([input["bev_map"] for input in batched_inputs]).to(self.device).flatten(3)
        tangent_init = torch.stack([input["tangent_init"] for input in batched_inputs]).to(self.device)
        distance_init = torch.stack([input["distance_init"] for input in batched_inputs]).to(self.device)
        image = torch.cat([bev_map, tangent_init, distance_init[..., None]], dim=-1)
        image = image.permute(0, 3, 1, 2)
        # from common.probing import feature_dist
        # feature_dist(image, nonzero_channels=[0,1,2])
        if self.training:
            # label = torch.stack([input["sem_seg"] for input in batched_inputs]).to(self.device)
            regression = torch.stack([input["regression"] for input in batched_inputs]).to(self.device)
            regression2 = torch.stack([input["regression2"] for input in batched_inputs]).to(self.device)
            regression3 = torch.stack([input["regression3"] for input in batched_inputs]).to(self.device)
            regression4 = torch.stack([input["regression4"] for input in batched_inputs]).to(self.device)
            label_regression = torch.cat([regression[..., None], regression2, regression3[..., None], regression4[..., None]], dim=-1).permute(0, 3, 1, 2)
            training_mask = torch.stack([input["training_mask"] for input in batched_inputs]).to(self.device)
        features, mask_features = self.backbone(image)
        if len(features) == 1:
            features = features[0]
        if len(mask_features) == 1:
            mask_features = mask_features[0]
        if self.training:
            results, loss = self.sem_seg_head(features, mask_features, targets=None, targets_regressor=label_regression, targets_regressor_mask=training_mask, tangent_init_polar=tangent_init_polar)
            # from common.probing import feature_dist
            # feature_dist(results)
            self.training_input = batched_inputs
            self.training_inference = self.inference(batched_inputs, results)
            return loss
        else:
            results, _ = self.sem_seg_head(features, mask_features)
            # from common.probing import feature_dist
            # feature_dist(results)
            return self.inference(batched_inputs, results)

    def inference(self, batched_inputs, results):
        processed_results = []
        for result, input_per_image in zip(results, batched_inputs):
            processed_results.append({"regression": result})
        return processed_results


def build_directer(cfg):
    nclasses = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES

    from model.backbones import SalsaNext
    backbone = SalsaNext(custom_padding=cfg.MODEL.SEM_SEG_HEAD.CUSTOM_PADDING, multi_scale=cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS, deep_supervision=cfg.MODEL.SEM_SEG_HEAD.NUM_DEEP_LEVELS, in_channels=6, nclasses=nclasses, base_channels=32)

    return backbone
