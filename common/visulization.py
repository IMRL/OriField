import numpy as np
import cv2
from skimage import measure, color
from matplotlib import pyplot as plt

from .polar import *
from .peroidic import peroidic_np


class Visulizater:
    def __init__(self, colormap):
        self.colormap = colormap
        self.binarymap = np.array([[0.,0.,0.],[1.,1.,1.]])

    def tangent_vis(self, tangent_map, tangent_offset=None, tangent_mask=None, rescale = 12, thickness = 1, pltcm=None):
        tangent_map_length = (tangent_map[..., 0]**2+tangent_map[..., 1]**2)**.5
        if tangent_offset is None:
            tangent_offset = np.full(tangent_map.shape[:-1], 0.)
        
        offseted_map_polar = cross_add(tangent_map, tangent_offset)
        if pltcm is None:
            offseted_map_x, offseted_map_y = polar_to_cartesian(tangent_map_length, offseted_map_polar)
            offseted_map = np.stack([offseted_map_x, offseted_map_y], axis=-1)
            offseted_map_length = np.linalg.norm(offseted_map, axis=-1)
            # print("((offseted_map_length > 0) & (offseted_map_length < 1)).sum()", ((offseted_map_length > 0) & (offseted_map_length < 1)).sum())
            # print("(offseted_map_length == 0).sum()", (offseted_map_length == 0).sum())
            # print("(offseted_map_length == 1).sum()", (offseted_map_length == 1).sum())
            halfrescale = rescale // 2 - 1
            tangent_img = np.zeros((offseted_map.shape[0]*rescale, offseted_map.shape[1]*rescale))
            for i in range(offseted_map.shape[0]):
                for j in range(offseted_map.shape[1]):
                    tx = offseted_map[i, j, 0]
                    ty = offseted_map[i, j, 1]
                    
                    oo = (i*rescale, j*rescale)
                    dd = (i*rescale+int(halfrescale*tx), j*rescale+int(halfrescale*ty))
                    dd_opp = (i*rescale-int(halfrescale*tx), j*rescale-int(halfrescale*ty))
                    end = dd
                    end_opp = dd_opp
                    oo_uv = (oo[1], oo[0])
                    end_uv = (end[1], end[0])
                    end_opp_uv = (end_opp[1], end_opp[0])
                    cv2.arrowedLine(tangent_img, end_opp_uv, end_uv, (1, 1, 1), thickness)
            return tangent_img
        else:
            tangent_vis = self.colormap_vis((peroidic_np(offseted_map_polar, norm=True, abs=False) + 1) / 2, pltcm)
            tangent_vis[..., -1] = tangent_map_length
            if tangent_mask is not None:
                tangent_vis[~tangent_mask] = np.array([0,0,0,0.])
            return tangent_vis

    def movement_vis(self, movement_map, rescale = 12, thickness = 1):
        halfrescale = rescale // 2 - 1
        
        movement_img = np.zeros((movement_map.shape[0]*rescale, movement_map.shape[1]*rescale))
        for i in range(movement_map.shape[0]):
            for j in range(movement_map.shape[1]):
                move = movement_map[i, j]
                if move > 0:
                    oo = (i*rescale, j*rescale)
                    d0 = (i*rescale+halfrescale, j*rescale)
                    d1 = (i*rescale+halfrescale, j*rescale+halfrescale)
                    d2 = (i*rescale, j*rescale+halfrescale)
                    d3 = (i*rescale-halfrescale, j*rescale+halfrescale)
                    d4 = (i*rescale-halfrescale, j*rescale)
                    d5 = (i*rescale-halfrescale, j*rescale-halfrescale)
                    d6 = (i*rescale, j*rescale-halfrescale)
                    d7 = (i*rescale+halfrescale, j*rescale-halfrescale)
                    ds = [oo, d0, d1, d2, d3, d4, d5, d6, d7]
                    end = ds[1:][move-1]
                    end_opp = ds[1:][(move-1+4)%8]
                    oo_uv = (oo[1], oo[0])
                    end_uv = (end[1], end[0])
                    end_opp_uv = (end_opp[1], end_opp[0])
                    cv2.arrowedLine(movement_img, end_opp_uv, end_uv, (1, 1, 1), thickness)
        return movement_img

    def naive_vis(self, value):
        if value.ndim == 4:
            value = value[:, :, value.shape[2] // 2, :]
        return value

    def repeat_vis(self, value):
        return np.repeat(value[..., None], 3, axis=-1)

    def defined_label_vis(self, label):
        return self.colormap[label]

    def clustered_label_vis(self, label):
        return color.label2rgb(label, bg_label=0)

    def binary_label_vis(self, label):
        return self.binarymap[(label != 0).astype(np.int32)]

    def bidirection_vis(self, value):
        red = np.zeros(value.shape+(3,), dtype=np.float32)
        red[..., 0] = 1
        yellow = np.zeros(value.shape+(3,), dtype=np.float32)
        yellow[..., 0] = 1
        yellow[..., 1] = 1
        return np.where(value[..., None] < 0, red * (-value[..., None]), yellow * (value[..., None]))

    def colormap_vis(self, value, pltcm):
        colors = plt.get_cmap(pltcm)(value)
        return colors
    
    def merge_layer(self, colored_bg, colored_fg, fg_mask):
        result = colored_bg.copy()
        result[fg_mask] = colored_fg[fg_mask]
        return result

    def stack_layer(self, colored_bg, fg_mask_list, fg_color_list=None):
        # print("stack_layer", colored_bg.shape, len(fg_mask_list), len(fg_color_list))
        if not isinstance(fg_mask_list, list) and not isinstance(fg_mask_list, tuple):
            fg_mask_list = [fg_mask_list]
        if fg_color_list is None:
            fg_color_list = [np.array([1.]*colored_bg.shape[-1])]*len(fg_mask_list)
        elif not isinstance(fg_color_list, list) and not isinstance(fg_color_list, tuple):
            fg_color_list = [fg_color_list]*len(fg_mask_list)
        result = colored_bg.copy()
        for i in range(len(fg_mask_list)):
            fg_mask = fg_mask_list[i]
            fg_color = fg_color_list[i]
            result[fg_mask] = fg_color
        return result