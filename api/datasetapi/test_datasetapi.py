import numpy as np

from api.datasetapi.data_proxy import DataProxy


if __name__ == '__main__':
    DATASET_KEY = "k360"
    DATA_ROOT = "/home/yuminghuang/dataset/"
    data_proxy = DataProxy.getInstance(DATASET_KEY, DATA_ROOT)
    data_dict = data_proxy.val_salsa_loader[1000]

    pointcloud = data_dict["pointcloud"]
    sem_label = data_dict["sem_label"]
    proj_range_tensor = data_dict["proj_range_tensor"]

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(np.asarray(pointcloud[:, :3], dtype=np.float32))
    # pcd.colors = o3d.utility.Vector3dVector(np.asarray(data_proxy.val_salsa_loader.dataset.sem_color_lut[sem_label], dtype=np.float32))
    # o3d.visualization.draw_geometries([pcd])

    import matplotlib.pyplot as plt
    plt.imshow(proj_range_tensor)
    plt.show()
