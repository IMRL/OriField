import numpy as np
from scipy.ndimage import convolve
from scipy.spatial.distance import cdist
from scipy.spatial import KDTree
from .edt import replace_mask_with_nearest


def brocast_tangent(tangents_map, tangents_mask, brocase_mask=None, weight_by_distance=False):
    if brocase_mask is None:
        brocase_mask = np.full(tangents_mask.shape, True)
    if not weight_by_distance:
        brocasted_tangents_map = replace_mask_with_nearest(tangents_map, mask=~tangents_mask)
    else:
        brocasted_tangents_map, distance = replace_mask_with_nearest(tangents_map, mask=~tangents_mask, return_distance=True)
        brocasted_tangents_map *= np.exp(-distance/40)[..., None]
    return np.where(brocase_mask[..., None], brocasted_tangents_map, tangents_map)

def smooth_tangent(tangent_map, tangent_mask=None, kernel=np.ones((5,5))):
    if tangent_mask is None:
        tangent_mask = np.full(tangent_map.shape[:2], True)
    tangent_map_length = (tangent_map[..., 0] ** 2 + tangent_map[..., 1] ** 2) ** .5
    convolved_tangents_map_x = convolve(tangent_map[..., 0], kernel, mode='constant', cval=0)
    convolved_tangents_map_y = convolve(tangent_map[..., 1], kernel, mode='constant', cval=0)
    convolved_tangents_map_length = (convolved_tangents_map_x ** 2 + convolved_tangents_map_y ** 2) ** .5
    smoothed_tangents_map = np.where(tangent_mask[..., None] & (convolved_tangents_map_length[..., None] > 0), np.stack([
        convolved_tangents_map_x / convolved_tangents_map_length * tangent_map_length, 
        convolved_tangents_map_y / convolved_tangents_map_length * tangent_map_length,
    ], axis=-1), tangent_map)
    return smoothed_tangents_map

def smooth_and_brocast_tangent(tangents, tangents_mask, brocase_mask):
    smoothed_tangents = smooth_tangent(tangents, tangents_mask)
    brocasted_smoothed_tangents = brocast_tangent(
        smoothed_tangents, tangents_mask, 
        brocase_mask=brocase_mask,
        # weight_by_distance=True,
    )
    smoothed_brocasted_smoothed_tangents = smooth_tangent(brocasted_smoothed_tangents, brocase_mask)
    return smoothed_tangents, brocasted_smoothed_tangents, smoothed_brocasted_smoothed_tangents

def pierce_tangent(points, tangents, shape):
    tangent_result = np.full(shape+(tangents.shape[-1],), 0.)
    distance_result = np.full(shape, 0.)
    if points.shape[0] > 0 and tangents.shape[0] > 0:
        tangents_mask_points = np.mgrid[:shape[0], :shape[1]].reshape(2, -1).T
        # indices = cdist(tangents_mask_points, points).argmin(-1)
        kdtree = KDTree(points)
        distances, indices = kdtree.query(tangents_mask_points)
        tangent_result[tangents_mask_points.T[0], tangents_mask_points.T[1]] = tangents[indices]
        distance_result[tangents_mask_points.T[0], tangents_mask_points.T[1]] = distances
    return tangent_result, distance_result

def normalize_distance(distances, img_h, img_w):
    return 1 - distances / (min(img_h, img_w) / 2)
