import os
from tqdm import tqdm
from multiprocessing import Pool
from dataloader.dataset_loader import valset_loader
from common.traversibility import estimate_frontier_and_skeleton
from common import planning


DATASET_TYPE = os.getenv('DATASET_TYPE')
assert DATASET_TYPE in ["semantic-kitti", "um-osm-planner"]
if DATASET_TYPE == "semantic-kitti":
    planner_skeleton_scoring = planning.SkeletonScoring(400, 400, 0.16, {'Left': -32, 'Right': 32, 'Front': 32, 'Back': -32, 'Bottom': -2.5, 'Top': 3.5})
elif DATASET_TYPE == "um-osm-planner":
    planner_skeleton_scoring = planning.SkeletonScoring(224, 224, 0.16, {'Left': -17.92, 'Right': 17.92, 'Front': 17.92, 'Back': -17.92, 'Bottom': -2.5, 'Top': 3.5})


def process_index(index):
    data_dict = valset_loader[index]

    label_padded = data_dict["label_padded"]
    binary_traversable = data_dict["binary_traversable"]
    greedy_points = data_dict["greedy_points"]
    greedy_tangents = data_dict["greedy_tangents"]
    label_mask = data_dict["label_mask"]
    binary_free = label_mask.copy()

    binary_frontier, cluster_frontier, label_pollute, baised_skeleton, baised_frontier_skeleton, skeleton_binary_traversable, skeleton_cluster_frontier = \
        estimate_frontier_and_skeleton(label_padded, binary_traversable)

    seq_id, frame_id = valset_loader.dataset_parsePathInfoByIndex(index)
    pose = valset_loader.dataset_poses[seq_id][int(frame_id)]

    planner_skeleton_scoring.prepare(frame_id, pose, skeleton_cluster_frontier, greedy_points, greedy_tangents, binary_traversable, os.path.join(valset_loader.dataset_root.replace('semantic-kitti', "bev-kitti"), seq_id))
    point, path = planner_skeleton_scoring.plan(frame_id, os.path.join(valset_loader.dataset_root.replace('semantic-kitti', "bev-kitti"), seq_id))
    print("point, path", point, path)


if __name__ == "__main__":
    total = list(range(0, len(valset_loader)))

    n_procces = 1
    with Pool(n_procces) as p:
        ret_list = list(tqdm(p.imap_unordered(process_index, total), total=len(total)))