# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN

def add_model_config(cfg):
    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    # backbone
    cfg.MODEL.SEM_SEG_HEAD.CUSTOM_PADDING = False
    cfg.MODEL.SEM_SEG_HEAD.NUM_FEATURE_LEVELS = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_DEEP_LEVELS = 1

    # head
    cfg.MODEL.SEM_SEG_HEAD.DIRECT_PRED = False

    cfg.MODEL.SEM_SEG_HEAD.NO_FOCAL = False
    cfg.MODEL.SEM_SEG_HEAD.CLASS_BALANCE = False
    cfg.MODEL.SEM_SEG_HEAD.LOVASZ = False
    cfg.MODEL.SEM_SEG_HEAD.LOVASZ_WEIGHT = 1.0
    cfg.MODEL.SEM_SEG_HEAD.BOUNDARY = False
    cfg.MODEL.SEM_SEG_HEAD.BOUNDARY_WEIGHT = 1.0
