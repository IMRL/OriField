import numpy as np
from scipy.spatial.distance import directed_hausdorff
from .circle_intersect import find_intersections


def l2_distance(traj, traj_ref):
    metric = {}
    d = np.linalg.norm(traj - traj_ref, axis=1)
    metric['ADE'] = np.mean(d)
    metric['FDE'] = d[-1]
    metric['MAX'] = np.max(d)
    return metric

def interp_l2_distance_3(traj, traj_ref, key_meters, samples=20):
    traj_interp = find_intersections(traj, np.array([0., 0.]), key_meters, samples)
    traj_interp_ref = find_intersections(traj_ref, np.array([0., 0.]), key_meters, samples)
    valid_mask = ~np.isnan(traj_interp).any(1) & ~np.isnan(traj_interp_ref).any(1)
    traj_interp, traj_interp_ref = traj_interp[valid_mask], traj_interp_ref[valid_mask]
    
    metric = l2_distance(traj_interp, traj_interp_ref)
    return traj_interp, traj_interp_ref, metric

def hausdorff_dist(traj1, traj2):
    """
    Double side hausdorff distance
    """
    dist1 = directed_hausdorff(traj1, traj2)[0]
    dist2 = directed_hausdorff(traj2, traj1)[0]
    return max(dist1, dist2)
