import numpy as np
import cv2


class Convertor:
    def __init__(self, img_h=None, img_w=None, res=None, lidar_range=None) -> None:
        self.img_h, self.img_w, self.res, self.lidar_range = img_h, img_w, res, lidar_range

    def reinit(self, img_h=None, img_w=None, res=None, lidar_range=None) -> None:
        self.img_h, self.img_w, self.res, self.lidar_range = img_h, img_w, res, lidar_range

    def img2car(self, img_points, normalized=False):
        """
        Convert coordinates from image space to ego-car
        """
        if normalized:
            img_points = np.stack([img_points[:, 0] * self.img_w, img_points[:, 1] * self.img_h], axis=-1)
        car_points = np.stack([
            img_points[:, 0] * self.res + self.lidar_range['Left'],
            self.lidar_range['Front'] - img_points[:, 1] * self.res,
        ], axis=-1)
        return car_points

    def matrix2img(self, matrix_points):
        return np.stack([matrix_points[:, 1], matrix_points[:, 0]], axis=-1)

    def matrix2car(self, matrix_points):
        """
        Convert coordinates from matrix space to ego-car
        """
        return self.img2car(self.matrix2img(matrix_points))

    def car2img(self, car_point, normalize=False):
        """
        Convert coordinates from ego-car to image space
        """
        img_points = np.stack([
            (car_point[:, 0] - self.lidar_range['Left']) / self.res,
            (self.lidar_range['Front'] - car_point[:, 1]) / self.res
        ], axis=-1).astype(np.int32)
        if normalize:
            img_points = np.stack([img_points[:, 0] / self.img_w, img_points[:, 1] / self.img_h], axis=-1)
        return img_points

    def img2matrix(self, img_points):
        return np.stack([img_points[:, 1], img_points[:, 0]], axis=-1)

    def car2matrix(self, car_points):
        """
        Convert coordinates from ego-car space to matrix
        """
        return self.img2matrix(self.car2img(car_points))

    def matrix_list2map(self, list, node=True, edge=False, thickness=1):
        if node:
            # node_map = np.full((self.img_h, self.img_w), False)
            # node_map[list[:, 0], list[:, 1]] = True
            node_map = np.full((self.img_h, self.img_w), 0.)
            for i in range(0, len(list)):
                child_node = list[i]
                child_node_uv = (child_node[1], child_node[0])
                cv2.circle(node_map, child_node_uv, radius=0, color=(1, 1, 1), thickness=-1)
            node_map = node_map.astype(np.bool)
        if edge:
            edge_map = np.full((self.img_h, self.img_w), 0.)
            for i in range(1, len(list)):
                current_node = list[i-1]
                child_node = list[i]
                current_node_uv = (current_node[1], current_node[0])
                child_node_uv = (child_node[1], child_node[0])
                cv2.line(edge_map, current_node_uv, child_node_uv, (1, 1, 1), thickness)
            edge_map = edge_map.astype(np.bool)
        if node and not edge:
            return node_map
        elif not node and edge:
            return edge_map
        elif node and edge:
            return node_map, edge_map
        
    def matrix_point2map(self, point, radius=1):
        map = np.full((self.img_h, self.img_w), False)
        xx, yy = np.ogrid[:self.img_h, :self.img_w]
        circle = (xx - point[0])**2 + (yy - point[1])**2 <= radius**2
        map[circle] = True
        return map
