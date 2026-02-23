import os
import numpy as np

from .utils import get_panoptic_weight_alpha, get_w1_w2_long_tail_sem

from .dataset import pc_processor

from nuscenes.utils.data_io import load_bin_file
# from nuscenes.lidarseg.class_histogram import get_lidarseg_num_points_per_class, get_panoptic_num_instances_per_class


class Nuscenes(pc_processor.dataset.nuScenes.Nuscenes):
    def __init__(self,
                 panoptic,
                 root,
                 version='v1.0-trainval',
                 split='train',
                 return_ref=False,
                 has_image=True,
                 has_pcd=True,
                 has_label=True):
        super(Nuscenes, self).__init__(root=root,
                 version=version,
                 split=split,
                 return_ref=return_ref,
                 has_image=has_image,
                 has_pcd=has_pcd,
                 has_label=has_label)

        print("[panoptic: {}]".format(panoptic))

        # addition
        self.data_config = self.nusc
        self.class_map_lut_inv = np.zeros((len(self.mapped_cls_name)), dtype=np.int32)
        for i in range(0, len(self.mapped_cls_name)):
            self.class_map_lut_inv[i] = i
        self.sem_color_lut = self.getColorMap()

        # meta
        stuff_classes = ["ignore",
                            "barrier", "bicycle", "bus", "car", "construction_vehicle", "motorcycle", "pedestrian",
                            "traffic_cone", "trailer", "truck",
                            "driveable_surface", "other_flat", "sidewalk", "terrain", "manmade", "vegetation"]
        thing_dataset_id_to_contiguous_id = {k: k for i, k in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])}
        stuff_dataset_id_to_contiguous_id = {k: k for i, k in
                                                enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])}
        self.thing_dataset_id_to_contiguous_id = thing_dataset_id_to_contiguous_id
        self.stuff_dataset_id_to_contiguous_id = stuff_dataset_id_to_contiguous_id
        self.stuff_classes = stuff_classes
        self.ignore_label = 0
        if not panoptic:
            self.evaluator_type = "sem_seg"
        else:
            self.evaluator_type = "ade20k_panoptic_seg"

        # weight
        # num_points_per_class = get_lidarseg_num_points_per_class(self.nusc, sort_by='count_asc')
        num_points_per_class = {'vehicle.emergency.ambulance': 2218, 'animal': 5385, 'human.pedestrian.personal_mobility': 8723,
                                'human.pedestrian.stroller': 8809, 'human.pedestrian.police_officer': 9159, 'human.pedestrian.child': 9655,
                                'human.pedestrian.wheelchair': 12168, 'vehicle.emergency.police': 59590, 'movable_object.debris': 66861,
                                'human.pedestrian.construction_worker': 139443, 'vehicle.bicycle': 141351, 'static_object.bicycle_rack': 163126,
                                'vehicle.bus.bendy': 357463, 'vehicle.motorcycle': 427391, 'movable_object.pushable_pullable': 718641,
                                'movable_object.trafficcone': 736239, 'static.other': 817150, 'vehicle.construction': 1514414, 'noise': 2061156,
                                'human.pedestrian.adult': 2156470, 'vehicle.bus.rigid': 4247297, 'vehicle.trailer': 4907511, 'flat.other': 8559216,
                                'movable_object.barrier': 9305106, 'vehicle.truck': 15841384, 'vehicle.car': 38104219, 'flat.sidewalk': 70197461,
                                'flat.terrain': 70289730, 'static.vegetation': 122581273, 'static.manmade': 178178063, 'flat.driveable_surface': 316958899,
                                'vehicle.ego': 337070621}
        print("num_points_per_class", num_points_per_class)
        num_points_total = 0
        for k, v in num_points_per_class.items():
            num_points_total += v
        cls_content = {}
        for k, v in num_points_per_class.items():
            cls_content[k] = v / num_points_total
        print("cls_content", cls_content)
        content = np.zeros(len(self.mapped_cls_name), dtype=np.float32)
        for cl, freq in cls_content.items():
            x_cl = self.map_name_from_general_index_to_segmentation_index[self.nusc.lidarseg_name2idx_mapping[cl]]
            content[x_cl] += freq
        self.cls_freq = content
        print("self.cls_freq", self.cls_freq)

        # num_instances_per_class = get_panoptic_num_instances_per_class(self.nusc, sort_by='count_asc')
        # print("num_instances_per_class", num_instances_per_class)
        # num_frames_total = self.__len__()
        # print("num_frames_total", num_frames_total)
        # ins_content = {}
        # for k, v in num_instances_per_class.items():
        #     ins_content[k] = v / num_frames_total
        # print("ins_content", ins_content)
        # reweight = np.zeros(len(self.mapped_cls_name), dtype=np.float32)
        # for cl, freq in ins_content.items():
        #     x_cl = self.map_name_from_general_index_to_segmentation_index[self.nusc.lidarseg_name2idx_mapping[cl]]
        #     reweight[x_cl] += freq
        reweight = [1, 12.01406952, 1.422138837, 1.375163399, 11.14858231, 1.680576504, 1.480326383, 7.065445731, 5.475400723, 2.578805568, 3.099755042, 1, 1, 1, 1, 1, 1]
        print("reweight", reweight)

        cls_weight = 1 / (self.cls_freq + 1e-3)
        cls_weight[0] = 0  # ignore
        cls_alpha = np.log(1 + cls_weight)
        cls_alpha = cls_alpha / cls_alpha.max()
        cls_weight_panoptic, cls_alpha_panoptic = get_panoptic_weight_alpha(cls_weight, reweight)

        self.panoptic = panoptic

        if not self.panoptic:
            self.cls_weight = cls_weight
            self.cls_alpha = cls_alpha
        else:
            self.cls_weight = cls_weight_panoptic
            self.cls_alpha = cls_alpha_panoptic
        self.long_tail = 0.1
        self.w1, self.w2, self.long_tailed_sem = get_w1_w2_long_tail_sem(self.panoptic, self.long_tail,
                                                                         cls_weight, cls_alpha,
                                                                         cls_weight_panoptic, cls_alpha_panoptic)

    def get_frame_name(self, index):
        return self.token_list[index]

    # def loadDataByIndex(self, index):
    #     pointcloud, sem_label, inst_label = super(Nuscenes, self).loadDataByIndex(index)
    #     return pointcloud, sem_label.squeeze(1), inst_label

    def loadDataByIndex(self, index):
        if self.has_image:
            lidar_sample_token = self.token_list[index]['lidar_token']
        else:
            lidar_sample_token = self.token_list[index]

        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            if not self.panoptic:
                annotated_data = np.expand_dims(
                    np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            else:
                panoptic_label_arr = np.zeros_like(raw_data[:, 0], dtype=int)
        else:
            if not self.panoptic:
                lidarseg_path = os.path.join(self.data_path,
                                             self.nusc.get('lidarseg', lidar_sample_token)['filename'])
                annotated_data = np.fromfile(
                    lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label
            else:
                panoptic_path = os.path.join(self.data_path,
                                             self.nusc.get('panoptic', lidar_sample_token)['filename'])
                panoptic_label_arr = load_bin_file(panoptic_path, 'panoptic')
        pointcloud = raw_data[:, :4]
        if not self.panoptic:
            sem_label = annotated_data.squeeze(1)
            inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        else:
            sem_label = panoptic_label_arr // 1000
            inst_label = panoptic_label_arr % 1000
        return pointcloud, sem_label, inst_label

    def labelMapping(self, sem_label):
        # print("self.map_name_from_general_index_to_segmentation_index", self.map_name_from_general_index_to_segmentation_index)
        sem_label = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(
            sem_label)  # n
        return sem_label
