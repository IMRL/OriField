import numpy as np
from numba import jit


def inverse_extrinsic(extrinsic, pointcloud):
    # [B, 3, 3]
    rotation = extrinsic[..., 0:3, 0:3]
    # translation [B, 1, 3]
    translation = extrinsic[..., 0:3, 3][:, None]

    # To vehicle frame.
    # [B, L, 3]
    xyz = pointcloud[:, :, :3]
    xyz = np.einsum('bkr,blr->blk', np.linalg.inv(rotation), xyz - translation)
    ret = pointcloud.copy()
    ret[:, :, :3] = xyz
    return ret


def azimuth_inclination_slot(height, width, extrinsic, inclination):
    # [B].
    az_correction = np.arctan2(extrinsic[..., 1, 0], extrinsic[..., 0, 0]) / (2 * np.pi)
    # [W].
    ratios = (np.arange(width, 0, -1) - .5) / width
    # [B, W].
    azimuth_slot = (ratios[None] - az_correction[:, None] + 1) % 1
    azimuth_slot = (azimuth_slot * 2. - 1.) * np.pi
    # [B, H]
    inclination_slot = inclination[:, ::-1]

    return azimuth_slot, inclination_slot


def index_azimuth_inclination(azimuth_slot, inclination_slot, pointcloud):
    x = pointcloud[:, :, 0]
    y = pointcloud[:, :, 1]
    z = pointcloud[:, :, 2]
    depth = np.linalg.norm(pointcloud[:, :, :3], axis=-1)  # B L
    sin_incl = z / depth
    inclination = np.arcsin(sin_incl)
    p = depth * np.cos(inclination)
    cos_azimuth = x / p
    cos_azimuth = np.clip(cos_azimuth, -1, 1)
    azimuth = np.arccos(cos_azimuth)
    azimuth = np.where(y < 0, -azimuth, azimuth)

    azimuth_idx = index_azimuth_jit(azimuth, azimuth_slot)
    inclination_idx = index_inclination_jit(inclination, inclination_slot)
    return azimuth_idx, inclination_idx


@jit(nopython=True)
def index_azimuth_jit(azimuth, azimuth_slot):  # B L, B W -> B L
    index = np.zeros(azimuth.shape, dtype=np.int64)
    for i in range(azimuth.shape[0]):
        slot = azimuth_slot[i]
        for j in range(azimuth.shape[1]):
            p = azimuth[i, j]
            idx = -1
            for s in range(len(slot)):
                left = slot[s-1]
                middle = slot[s]
                # right = slot[s+1]  # overflow!
                if left >= middle:
                    if left >= p:
                        if p > middle:
                            if left - p < p - middle:
                                idx = s-1
                            else:
                                idx = s
                            break
                else:
                    if p <= left:
                        idx = s-1
                        break
                    elif p >= middle:
                        idx = s
                        break
            index[i, j] = idx
    return index


@jit(nopython=True)
def index_inclination_jit(inclination, inclination_slot):  # B L, B H -> B L
    index = np.zeros(inclination.shape, dtype=np.int64)
    for i in range(inclination.shape[0]):
        slot = inclination_slot[i]
        slot_left = slot[0]
        slot_right = slot[-1]
        for j in range(inclination.shape[1]):
            p = inclination[i, j]
            idx = -1
            for s in range(len(slot)):
                left = slot[s-1]
                middle = slot[s]
                # right = slot[s+1]  # overflow!
                if left >= middle:
                    if left >= p:
                        if p > middle:
                            if left - p < p - middle:
                                idx = s-1
                            else:
                                idx = s
                            break
                else:
                    if p <= left:
                        idx = s-1
                        break
                    elif p >= middle:
                        idx = s
                        break
            index[i, j] = idx
    return index


def project(pointcloud, height, width, extrinsic, inclination):
    pointcloud = inverse_extrinsic(extrinsic, pointcloud)
    azimuth_slot, inclination_slot = azimuth_inclination_slot(height, width, extrinsic, inclination)
    # print("azimuth_slot, inclination_slot", azimuth_slot, inclination_slot)
    azimuth_idx, inclination_idx = index_azimuth_inclination(azimuth_slot, inclination_slot, pointcloud)
    range_image = np.zeros(inclination_slot.shape[:2]+(azimuth_slot.shape[1],pointcloud.shape[2]))
    range_image[np.arange(inclination_idx.shape[0])[None], inclination_idx, azimuth_idx] = pointcloud
    return range_image


class RangeProjection(object):
    def __init__(self, proj_h, proj_w):
        self.proj_h = proj_h
        self.proj_w = proj_w
        self.cached_data = {}

    def doProjection(self, pointcloud: np.ndarray, extrinsic: np.ndarray, inclination: np.ndarray):
        self.cached_data = {}

        pointcloud = inverse_extrinsic(extrinsic[None], pointcloud[None]).squeeze(0)
        azimuth_slot, inclination_slot = azimuth_inclination_slot(self.proj_h, self.proj_w, extrinsic[None], inclination[None])
        azimuth_slot, inclination_slot = azimuth_slot.squeeze(0), inclination_slot.squeeze(0)
        # print("azimuth_slot, inclination_slot", azimuth_slot, inclination_slot)
        azimuth_idx, inclination_idx = index_azimuth_inclination(azimuth_slot[None], inclination_slot[None], pointcloud[None])
        azimuth_idx, inclination_idx = azimuth_idx.squeeze(0), inclination_idx.squeeze(0)
        # print("azimuth_idx, inclination_idx", azimuth_idx, inclination_idx)

        # get depth of all points
        depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)

        proj_x = azimuth_idx
        proj_y = inclination_idx

        self.cached_data["uproj_x_idx"] = proj_x.copy()
        self.cached_data["uproj_y_idx"] = proj_y.copy()
        self.cached_data["uproj_depth"] = depth.copy()

        # order in decreasing depth
        indices = np.arange(depth.shape[0])
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        indices = indices[order]
        pointcloud = pointcloud[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        # get projection result
        proj_range = np.full((self.proj_h, self.proj_w), -1, dtype=np.float32)
        proj_range[proj_y, proj_x] = depth

        proj_pointcloud = np.full(
            (self.proj_h, self.proj_w, pointcloud.shape[1]), -1, dtype=np.float32)
        proj_pointcloud[proj_y, proj_x] = pointcloud

        proj_idx = np.full((self.proj_h, self.proj_w), -1, dtype=np.int32)
        proj_idx[proj_y, proj_x] = indices

        proj_mask = (proj_idx > 0).astype(np.int32)

        return proj_pointcloud, proj_range, proj_idx, proj_mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    pointcloud = np.random.rand(1, 100, 4)
    extrinsic = np.eye(4)[None]
    inclination = np.arange(0, 0.64, 0.01)[None]
    range_image = project(pointcloud, 64, 100, extrinsic, inclination)
    plt.imshow(np.linalg.norm(range_image[0], axis=-1))
    plt.show()
    proj_pointcloud, proj_range, proj_idx, proj_mask = RangeProjection(64, 100).doProjection(pointcloud[0], extrinsic[0], inclination[0])
    plt.imshow(proj_range)
    plt.show()

    from dataset import Waymo
    data_path = '/home/yuminghuang/dataset/waymo-mini'
    dataset = Waymo(root=data_path, version='v2.0.0', split='train', has_image=False, has_label=False)
    pointcloud, sem_label, inst_label = dataset.loadDataByIndex(0)
    extrinsic, inclination = dataset.loadCalibrationByIndex(0)
    proj_pointcloud, proj_range, proj_idx, proj_mask = RangeProjection(64, 2650).doProjection(pointcloud, extrinsic, inclination)
    plt.imshow(proj_range)
    plt.show()

    dataset = Waymo(root=data_path, version='v2.0.0', split='val', has_image=False, has_label=False)
    pointcloud, sem_label, inst_label = dataset.loadDataByIndex(0)
    extrinsic, inclination = dataset.loadCalibrationByIndex(0)
    proj_pointcloud, proj_range, proj_idx, proj_mask = RangeProjection(64, 2650).doProjection(pointcloud, extrinsic, inclination)
    plt.imshow(proj_range)
    plt.show()

    dataset = Waymo(root=data_path, version='v2.0.0', split='test', has_image=False, has_label=False)
    pointcloud, sem_label, inst_label = dataset.loadDataByIndex(0)
    extrinsic, inclination = dataset.loadCalibrationByIndex(0)
    proj_pointcloud, proj_range, proj_idx, proj_mask = RangeProjection(64, 2650).doProjection(pointcloud, extrinsic, inclination)
    plt.imshow(proj_range)
    plt.show()
