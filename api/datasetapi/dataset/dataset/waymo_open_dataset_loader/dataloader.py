import numpy as np
import torch
from torch.utils.data import Dataset

from projection import RangeProjection


class WaymoLoader(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.projection = RangeProjection(64, 2650)
        self.proj_img_mean = torch.tensor([19.84, 1.73, 1.10, 1.08, 10.21], dtype=torch.float32)
        self.proj_img_stds = torch.tensor([17.36, 19.43, 16.99, 1.30, 377.19], dtype=torch.float32)

    def __getitem__(self, index):
        pointcloud, sem_label, inst_label = self.dataset.loadDataByIndex(index)
        extrinsic, inclination = self.dataset.loadCalibrationByIndex(index)
        proj_pointcloud, proj_range, proj_idx, proj_mask = self.projection.doProjection(pointcloud, extrinsic, inclination)

        proj_mask_tensor = torch.from_numpy(proj_mask)
        mask = proj_idx > 0

        proj_idx_tensor = torch.from_numpy(proj_idx)
        proj_idx_tensor = proj_idx_tensor * proj_mask_tensor

        proj_sem_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        proj_sem_label[mask] = sem_label[proj_idx[mask]]
        proj_sem_label_tensor = torch.from_numpy(proj_sem_label)
        proj_sem_label_tensor = proj_sem_label_tensor * proj_mask_tensor.float()

        proj_inst_label = np.zeros((proj_mask.shape[0], proj_mask.shape[1]), dtype=np.float32)
        proj_inst_label[mask] = inst_label[proj_idx[mask]]
        proj_inst_label_tensor = torch.from_numpy(proj_inst_label)
        proj_inst_label_tensor = proj_inst_label_tensor * proj_mask_tensor.float()

        proj_range_tensor = torch.from_numpy(proj_range)
        proj_xyz_tensor = torch.from_numpy(proj_pointcloud[..., :3])
        proj_intensity_tensor = torch.from_numpy(proj_pointcloud[..., 3])
        proj_intensity_tensor = proj_intensity_tensor.ne(-1).float() * proj_intensity_tensor
        proj_feature_tensor = torch.cat(
            [proj_range_tensor.unsqueeze(0), proj_xyz_tensor.permute(2, 0, 1), proj_intensity_tensor.unsqueeze(0)], 0)

        proj_feature_tensor = (proj_feature_tensor - self.proj_img_mean[:, None, None]) / self.proj_img_stds[:, None, None]
        proj_feature_tensor = proj_feature_tensor * proj_mask_tensor.unsqueeze(0).float()

        uproj_x_tensor = torch.from_numpy(self.projection.cached_data["uproj_x_idx"]).long()
        uproj_y_tensor = torch.from_numpy(self.projection.cached_data["uproj_y_idx"]).long()
        uproj_depth_tensor = torch.from_numpy(self.projection.cached_data["uproj_depth"]).float()

        data_dict = {}
        data_dict["proj_feature_tensor"] = proj_feature_tensor
        data_dict["proj_range_tensor"] = proj_range_tensor
        data_dict["proj_mask_tensor"] = proj_mask_tensor
        data_dict["proj_idx_tensor"] = proj_idx_tensor
        data_dict["proj_sem_label_tensor"] = proj_sem_label_tensor
        data_dict["proj_inst_label_tensor"] = proj_inst_label_tensor

        data_dict["uproj_x_tensor"] = uproj_x_tensor
        data_dict["uproj_y_tensor"] = uproj_y_tensor
        data_dict["uproj_depth_tensor"] = uproj_depth_tensor

        data_dict["pointcloud"] = pointcloud
        data_dict["sem_label"] = sem_label
        data_dict["inst_label"] = inst_label

        return data_dict

    def __len__(self):
        return len(self.dataset)


if __name__ == '__main__':
    from dataset import Waymo
    data_path = '/home/yuminghuang/dataset/waymo-mini'
    dataset = Waymo(root=data_path, version='v2.0.0', split='train', has_image=False, has_label=True)
    dataloader = WaymoLoader(dataset)
    data_dict = dataloader[0]

    import matplotlib.pyplot as plt
    plt.imshow(data_dict["proj_range_tensor"])
    plt.show()
    plt.imshow(dataloader.dataset.sem_color_lut[data_dict["proj_sem_label_tensor"].numpy().astype(np.int32)])
    plt.show()
