import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import convolve
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.spatial.distance import cdist
from skimage import measure, color
from skimage.morphology import medial_axis, skeletonize
from skimage.graph import route_through_array
from skimage.draw import line
import networkx as nx
import rdp
from numba import jit
import matplotlib.pyplot as plt

from . import planning


def custom_medial_axis(binary_image, cluster_frontier):
    skeleton, distance = medial_axis(binary_image, return_distance=True)
    dist_on_skel = distance * skeleton
    cluster_frontier_skeleton = locate_skeleton_frontier(skeleton, cluster_frontier)
    desc = {
        "binary_image": binary_image,
        "cluster_frontier": cluster_frontier,
        "skeleton": skeleton,
        "cluster_frontier_skeleton": cluster_frontier_skeleton,
    }
    # fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(desc["binary_image"])
    # ax[1].imshow(desc["cluster_frontier"])
    # ax[2].imshow(desc["skeleton"])
    # ax[3].imshow(desc["cluster_frontier_skeleton"])
    # plt.show()

    pruned_skeleton, baised_skeleton, baised_frontier_skeleton = prune_skeletion(skeleton, cluster_frontier_skeleton)
    reduced_skeleton = pruned_skeleton.copy()
    reduced_cluster_frontier_skeleton = cluster_frontier_skeleton.copy()
    # reduced_grid, reduced_rdp, _ = reduce_skeleton(pruned_skeleton)
    # reduced_skeleton = reduced_grid.copy()
    # reduced_cluster_frontier_skeleton = cluster_frontier_skeleton.copy()
    # reduced_cluster_frontier_skeleton[~reduced_skeleton] = 0
    # fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(binary_image)
    # ax[1].imshow(cluster_frontier)
    # ax[2].imshow(skeleton)
    # ax[3].imshow(cluster_frontier_skeleton)
    # ax[4].imshow(pruned_skeleton)
    # ax[5].imshow(reduced_skeleton)
    # ax[6].imshow(reduced_cluster_frontier_skeleton)
    # plt.show()

    return baised_skeleton, baised_frontier_skeleton, reduced_skeleton, reduced_cluster_frontier_skeleton


def custom_skeletonize(binary_image, cluster_frontier, method=None):
    skeleton = skeletonize(binary_image, method=method).astype(np.bool)
    cluster_frontier_skeleton = locate_skeleton_frontier(skeleton, cluster_frontier)

    # kernel = np.array([[1, 1, 1],
    #                 [1, 10, 1],
    #                 [1, 1, 1]])
    # neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    # endpoints = np.where(neighbor_count == 11)
    # endpoints = np.transpose(endpoints)
    # cluster_frontier_skeleton = np.zeros_like(cluster_frontier)
    # cluster_frontier_skeleton[endpoints.T[0], endpoints.T[1]] = np.arange(endpoints.shape[0])+1

    baised_skeleton = skeleton.copy()
    baised_frontier_skeleton = 0
    frontier_id_skeleton = np.unique(cluster_frontier_skeleton)[1:]
    for frontier_id in frontier_id_skeleton[np.random.permutation(frontier_id_skeleton.shape[0])][:1]:
        baised_frontier_skeleton = frontier_id

    reduced_skeleton = skeleton.copy()
    reduced_cluster_frontier_skeleton = cluster_frontier_skeleton.copy()

    return baised_skeleton, baised_frontier_skeleton, reduced_skeleton, reduced_cluster_frontier_skeleton


def locate_skeleton_frontier(skeleton, cluster_frontier):
    frontier_id = np.unique(cluster_frontier)[1:]
    cluster_frontier_skeleton = np.zeros_like(cluster_frontier)
    for fid in frontier_id:
        frontier = cluster_frontier == fid
        while True:
            frontierpoints = np.transpose(np.nonzero(frontier & skeleton))
            if len(frontierpoints) > 0:
                break
            else:
                frontier = binary_dilation(frontier, iterations=2)
        frontierpoint = frontierpoints[cdist([frontierpoints.mean(axis=0)], frontierpoints).argmin()]
        cluster_frontier_skeleton[tuple(frontierpoint)] = fid
    return cluster_frontier_skeleton


def prune_skeletion(skeleton, cluster_frontier_skeleton):
    # path_skeleton = np.zeros_like(skeleton)
    # cost_array = np.where(skeleton, 0, 1)
    # frontier_id_skeleton = np.unique(cluster_frontier_skeleton)[1:]
    # for i in range(len(frontier_id_skeleton)):
    #     for j in range(i, len(frontier_id_skeleton)):
    #         start_frontier_id = frontier_id_skeleton[i]
    #         end_frontier_id = frontier_id_skeleton[j]
    #         if start_frontier_id == end_frontier_id:
    #             continue  # Ignore path from a point to itself
    #         start_frontier = cluster_frontier_skeleton == start_frontier_id
    #         end_frontier = cluster_frontier_skeleton == end_frontier_id
    #         startpoint = np.transpose(np.nonzero(start_frontier))[0]
    #         endpoint = np.transpose(np.nonzero(end_frontier))[0]

    #         # fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
    #         # ax = axes.ravel()
    #         # ax[0].imshow(skeleton)
    #         # ax[1].imshow(start_frontier)
    #         # ax[2].imshow(end_frontier)
    #         # plt.show()
            
    #         # Find the path between start_point and end_point using the skeleton as a cost array
    #         # We invert the skeleton because route_through_array treats lower costs as more optimal paths
    #         timer2.put("route_through_array")
    #         path, _ = route_through_array(cost_array, start=tuple(startpoint), end=tuple(endpoint), fully_connected=True)
    #         timer2.get()

    #         timer2.put("map")
    #         path = np.array(list(map(list, path)))
    #         timer2.get()
    #         # Update the path_skeleton with the current path
    #         path_skeleton[path.T[0], path.T[1]] = True
    path_skeleton = np.zeros_like(skeleton)
    baised_skeleton = np.zeros_like(skeleton)
    baised_frontier_skeleton = 0
    dij_skeleton = planning.Dijkstra(skeleton)
    frontier_id_skeleton = np.unique(cluster_frontier_skeleton)[1:]
    for i, frontier_id in enumerate (frontier_id_skeleton[np.random.permutation(frontier_id_skeleton.shape[0])][:2]):
        movements_skeleton, tangents_skeleton = dij_skeleton.plan(cluster_frontier_skeleton == frontier_id)
        for endpoint in np.transpose(np.nonzero((cluster_frontier_skeleton != 0) & (cluster_frontier_skeleton != frontier_id))):
            current = tuple(endpoint)
            path_points, path_tangents = dij_skeleton.back_tracing(movements_skeleton, tangents_skeleton, current)
            path_skeleton[path_points.T[0], path_points.T[1]] = True
            if i == 0:
                baised_skeleton[path_points.T[0], path_points.T[1]] = True
        if i == 0:
            baised_frontier_skeleton = frontier_id
    # fig, axes = plt.subplots(4, 2, figsize=(8, 8), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(skeleton)
    # ax[1].imshow(cluster_frontier_skeleton)
    # ax[2].imshow(path_skeleton)
    # plt.show()
    return path_skeleton, baised_skeleton, baised_frontier_skeleton


@jit(nopython=True)
def get_next_nodes(skeleton, current_node, prev_node=None):
    img_h, img_w = skeleton.shape
    next_nodes = np.zeros((0, 2), dtype=np.int64)
    neighbors = [(current_node[0] + i, current_node[1] + j)
                for i in range(-1, 2)
                for j in range(-1, 2)
                if (i, j) != (0, 0)]
    for neighbor in neighbors:
        if neighbor[0] < 0 or neighbor[0] > img_h-1 or neighbor[1] < 0 or neighbor[1] > img_w-1:
            continue
        if prev_node is not None and neighbor == tuple(prev_node):
            continue
        if skeleton[neighbor]:
            next_nodes = np.append(next_nodes, np.array([[neighbor[0], neighbor[1]]]), axis=0)
    return next_nodes


@jit(nopython=True)
def trace_nodes(skeleton, nodes):
    edges_full = np.zeros((0, 4), dtype=np.int64)
    edges_key = np.zeros((0, 4), dtype=np.int64)
    edges_key_weight = np.zeros((0), dtype=np.float64)
    # Trace paths between nodes to create edges
    nodes_set = [(node[0], node[1]) for node in nodes]
    for node in nodes:
        next_nodes = get_next_nodes(skeleton, node)
        for next_node in next_nodes:
            prev_node = (node[0], node[1])
            current_node = (next_node[0], next_node[1])
            edges_full = np.append(edges_full, np.array([[prev_node[0], prev_node[1], current_node[0], current_node[1]]]), axis=0)  # must be [[]], not [][None]
            while current_node not in nodes_set:
                next_node = get_next_nodes(skeleton, current_node, prev_node)[0]
                prev_node = current_node
                current_node = (next_node[0], next_node[1])
                edges_full = np.append(edges_full, np.array([[prev_node[0], prev_node[1], current_node[0], current_node[1]]]), axis=0)
            edges_key = np.append(edges_key, np.array([[node[0], node[1], current_node[0], current_node[1]]]), axis=0)
            edges_key_weight = np.append(edges_key_weight, np.array([((node[0]-current_node[0])**2 + (node[1]-current_node[1])**2)**.5]))
    return edges_full, edges_key, edges_key_weight


def skeleton_to_graph(skeleton):
    # Define a convolution kernel to count neighbors
    kernel = np.array([[1, 1, 1],
                    [1, 10, 1],
                    [1, 1, 1]])
    # Convolve skeleton with the kernel to count neighbors
    neighbor_count = convolve(skeleton.astype(int), kernel, mode='constant', cval=0)
    # Define nodes: endpoints (one neighbor) or junctions (more than two neighbors)
    endpoints = np.where(neighbor_count == 11)
    junctions = np.where(neighbor_count > 12)
    nodes = np.concatenate([endpoints, junctions], axis=-1).T

    # Create graph from nodes
    G_full = nx.Graph()
    G_key = nx.Graph()

    if nodes.shape[0] > 0:
        edges_full, edges_key, edges_key_weight = trace_nodes(skeleton, nodes)
    else:
        # print("empty nodes")
        edges_full = np.zeros((0, 4), dtype=np.int64)
        edges_key = np.zeros((0, 4), dtype=np.int64)
        edges_key_weight = np.zeros((0), dtype=np.float64)
    G_full.add_edges_from(list(map(lambda x: ((x[0], x[1]), (x[2], x[3])), edges_full)))
    G_key.add_edges_from(list(map(lambda x: ((x[0], x[1]), (x[2], x[3]), {"weight": x[4]}), zip(*edges_key.T, edges_key_weight))))

    return G_full, G_key


def reduce_skeleton(skeleton):
    G_full, G_key = skeleton_to_graph(skeleton)

    G_grid = nx.Graph()
    G_rdp = nx.Graph()
    skeleton_grid = np.zeros_like(skeleton)
    skeleton_rdp = np.zeros_like(skeleton)
    # mst
    G_key = nx.minimum_spanning_tree(G_key, algorithm='kruskal')
    for u, v in G_key.edges():
        path_tuple = nx.shortest_path(G_full, source=u, target=v)

        path_array = np.array(list(map(list, path_tuple)))
        skeleton_grid[path_array.T[0], path_array.T[1]] = True
        # G_grid.add_edges_from(zip(path_tuple[:-1], path_tuple[1:]))

        # rdp_path_list = rdp.rdp(path_tuple, epsilon=1)
        # rdp_path_tuple = list(map(tuple, rdp_path_list))
        # # rdp_path_array = np.array(rdp_path_list)
        # # skeleton_rdp[rdp_path_array.T[0], rdp_path_array.T[1]] = True
        # for a, b in zip(rdp_path_tuple[:-1], rdp_path_tuple[1:]):
        #     rr, cc = line(*a, *b)
        #     skeleton_rdp[rr, cc] = True
        # G_rdp.add_edges_from(zip(rdp_path_tuple[:-1], rdp_path_tuple[1:]))

    # figure = plt.figure()
    # figure.add_subplot(2, 3, 1)
    # pos = {node: (node[1], -node[0]) for node in G.nodes()}  # Positional dict for nodes, flipping y-axis to match image orientation
    # nx.draw(G, pos, with_labels=True, node_size=300, node_color='red', font_size=8)
    # plt.gca().set_aspect('equal')
    # figure.add_subplot(2, 3, 4)
    # pos = {node: (node[1], -node[0]) for node in G2.nodes()}  # Positional dict for nodes, flipping y-axis to match image orientation
    # nx.draw(G2, pos, with_labels=True, node_size=300, node_color='red', font_size=8)
    # plt.gca().set_aspect('equal')

    # figure.add_subplot(2, 3, 2)
    # pos = {node: (node[1], -node[0]) for node in G_grid.nodes()}  # Positional dict for nodes, flipping y-axis to match image orientation
    # nx.draw(G_grid, pos, with_labels=True, node_size=300, node_color='red', font_size=8)
    # plt.gca().set_aspect('equal')
    # figure.add_subplot(2, 3, 5)
    # pos = {node: (node[1], -node[0]) for node in G_rdp.nodes()}  # Positional dict for nodes, flipping y-axis to match image orientation
    # nx.draw(G_rdp, pos, with_labels=True, node_size=300, node_color='red', font_size=8)
    # plt.gca().set_aspect('equal')
    
    # figure.add_subplot(2, 3, 3)
    # plt.imshow(skeleton_grid)
    # figure.add_subplot(2, 3, 6)
    # plt.imshow(skeleton_rdp)
    # plt.show()

    return skeleton_grid, skeleton_rdp, None
