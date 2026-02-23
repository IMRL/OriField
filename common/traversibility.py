import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import binary_erosion, binary_dilation
from skimage import measure, color
from .morphology import custom_medial_axis, custom_skeletonize
from .edt import replace_mask_with_nearest


def nearest_cluster(cluster, greedy_point, return_value=False):
    img_h, img_w = cluster.shape
    idx0 = np.full(cluster.shape, 1) * np.arange(cluster.shape[0])[:, None]
    idx1 = np.full(cluster.shape, 1) * np.arange(cluster.shape[1])[None, :]
    dis = np.where(
        cluster != 0, 
        ((greedy_point[0] - idx0) ** 2 + (greedy_point[1] - idx1) ** 2) ** .5, 
        np.full(cluster.shape, 1.) * (img_h + img_w)
    )
    ind = np.unravel_index(np.argmin(dis, axis=None), dis.shape)
    if not return_value:
        return cluster == cluster[ind]
    else:
        return cluster == cluster[ind], cluster[ind]


def padding_and_traversable(label, padding_distance_limit=10, clutter_size=40, flood=(0.5,0.5)):
    img_h, img_w = label.shape
    # label_erosion = label * binary_erosion(label, structure=np.ones((5,5)))
    label_padded = replace_mask_with_nearest(label, distance_limit=padding_distance_limit)
    _, _, binary_small = cluster_with_size(label_padded, clutter_size)
    label_erosion = label * ~binary_small# * binary_erosion(label, structure=np.ones((2,2)))
    label_padded = replace_mask_with_nearest(label_erosion, distance_limit=padding_distance_limit)
    label_segment = measure.label(label_padded, connectivity=1)
    binary_traversable = label_segment == label_segment[int(img_h*flood[0]), int(img_w*flood[1])]
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(label)
    # ax[1].imshow(label_erosion)
    # ax[2].imshow(label_padded)
    # ax[3].imshow(label_segment)
    # ax[4].imshow(binary_traversable)
    # plt.show()
    return label_erosion, label_padded, binary_traversable


def cluster_with_size(binary_or_label, size):
    cluster_all = measure.label(binary_or_label, connectivity=1)
    cluster_all_values, cluster_all_cnts = np.unique(cluster_all, return_counts=True)
    binary_small = np.isin(cluster_all, cluster_all_values[cluster_all_cnts < size])
    cluster_main = cluster_all.copy()
    cluster_main[binary_small] = 0
    return cluster_all, cluster_main, binary_small


def label_to_frontiers(label_padded, binary_traversable, boundary_mask, blind_size=160, frontier_size=40):
    binary_blind = label_padded == 0
    cluster_blind_all, cluster_blind_main, binary_blind_small = cluster_with_size(binary_blind, blind_size)
    label_padded_update = np.where(cluster_blind_main > 0, -cluster_blind_main, label_padded)  # -1: main, 0: small, >0: not blind
    label_pollute = replace_mask_with_nearest(label_padded_update, mask=binary_traversable)
    pollute = (label_pollute < 0) & binary_traversable
    boundary =  boundary_mask & binary_traversable
    binary_frontier = pollute | boundary
    cluster_frontier_all, cluster_frontier_main, binary_frontier_small = cluster_with_size(binary_frontier, frontier_size)
    # import matplotlib.pyplot as plt
    # fig, axes = plt.subplots(6, 2, figsize=(8, 8), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(label_padded)
    # ax[1].imshow(binary_blind)
    # ax[2].imshow(cluster_blind_main)
    # ax[3].imshow(binary_blind_small)
    # ax[4].imshow(label_padded_update)
    # ax[5].imshow(label_pollute)
    # ax[6].imshow(binary_traversable)
    # ax[7].imshow(pollute)
    # ax[8].imshow(boundary)
    # ax[9].imshow(binary_frontier)
    # ax[10].imshow(cluster_frontier_main)
    # ax[11].imshow(binary_frontier_small)
    # plt.show()
    return cluster_frontier_main > 0, cluster_frontier_main, label_pollute


def label_to_skeleton(binary_traversable, cluster_frontier):
    road_struct = binary_traversable

    baised_skeleton, baised_frontier_skeleton, skeleton, cluster_frontier_skeleton = custom_medial_axis(road_struct, cluster_frontier)
    # baised_skeleton, baised_frontier_skeleton, skeleton, cluster_frontier_skeleton = custom_skeletonize(road_struct, cluster_frontier, method='lee')
    return baised_skeleton, baised_frontier_skeleton, skeleton, cluster_frontier_skeleton


def estimate_frontier_and_skeleton(label_padded, binary_traversable):
    img_h, img_w = label_padded.shape

    boundary_pixels = 8
    h_idx = np.arange(img_h)
    w_idx = np.arange(img_w)
    h_mask = ((h_idx >= 0) & (h_idx < boundary_pixels)) | ((h_idx >= img_h-boundary_pixels) & (h_idx < img_h))
    w_mask = ((w_idx >= 0) & (w_idx < boundary_pixels)) | ((w_idx >= img_w-boundary_pixels) & (w_idx < img_w))
    boundary_mask = h_mask[:, None] | w_mask[None]

    binary_frontier, cluster_frontier, label_pollute = label_to_frontiers(label_padded, binary_traversable, boundary_mask)
    baised_skeleton, baised_frontier_skeleton, skeleton, cluster_frontier_skeleton = label_to_skeleton(binary_traversable, cluster_frontier)
    return binary_frontier, cluster_frontier, label_pollute, baised_skeleton, baised_frontier_skeleton, skeleton, cluster_frontier_skeleton
