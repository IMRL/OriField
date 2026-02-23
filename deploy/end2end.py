import os
import numpy as np
import torch
import onnxruntime as ort
from matplotlib import pyplot as plt
from .traceable import flatten_to_tuple


class End2End():
    def __init__(self, model_path, img_h, img_w, res, left, front, device='cuda'):
        self.img_h = img_h
        self.img_w = img_w
        self.res = res
        self.left = left
        self.front = front
        if device == 'cpu':
            self.ort_session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        else:
            self.ort_session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
        print("Inputs schema: %s", [item.name for item in self.ort_session.get_inputs()])
        print("Outputs schema: %s", [item.name for item in self.ort_session.get_outputs()])

    def eval_wrapper(self, inputs):
        inputs = self.basic_inputs(inputs)
        # assert len(inputs) == 1, "not support batch inference"
        flattened_inputs, _ = flatten_to_tuple(inputs)
        ort_inputs = {}
        for ort_input, flatten_input in zip(self.ort_session.get_inputs(), flattened_inputs):
            ort_inputs[ort_input.name] = flatten_input.cpu().numpy()
        # print("ort_inputs", ort_inputs)
        ort_outputs = self.ort_session.run(None, ort_inputs)
        flattened_outputs = [torch.tensor(item) for item in ort_outputs]
        outputs = {}
        outputs['pred_heatmap'] = flattened_outputs[0]
        outputs['pred_seg'] = flattened_outputs[1]
        outputs['pred_traj'] = flattened_outputs[2]
        return outputs

    def inference(self, inputs):
        self.load_data_to_gpu(inputs) 
        with torch.no_grad():
            pred_dicts = self.eval_wrapper(inputs)
            pred_dicts['pred_seg'] = pred_dicts['pred_seg'].cpu().numpy()
            if 'pred_heatmap' in pred_dicts:
                pred_dicts['pred_heatmap'] = pred_dicts['pred_heatmap'].cpu().numpy()
            pred_dicts['pred_traj'] = pred_dicts['pred_traj'].detach().cpu().numpy()

            # plt.imshow(np.concatenate([pred_dicts['pred_seg'][0], pred_dicts['pred_heatmap'][0]]))
            # plt.show()

            outputs = []
            for index in range(len(pred_dicts['pred_traj'])):
                pred_points = pred_dicts['pred_traj'][index]
                traj_pred_all = self.img2car(pred_points, normalized=True) # [num_points, 2]
                outputs.append(traj_pred_all)
            return outputs
    
    def img2car(self, img_points, normalized=True):
        """
        Convert coordinates from image space to ego-car
        """
        car_points = img_points.copy()
        if normalized:
            img_points[:,0] *= self.img_w
            img_points[:,1] *= self.img_h

        car_points[:,0] = img_points[:,0] * self.res + self.left
        car_points[:,1] = self.front - img_points[:,1] * self.res
        return car_points

    def car2img(self, car_point, normalize=False):
        """
        Convert coordinates from ego-car to image space
        """
        if car_point.ndim == 2: # ndarray
            img_points = car_point.copy()
            img_points[:,0] = (car_point[:,0] - self.lidar_range['Left']) / self.res
            img_points[:,1] = (self.lidar_range['Front'] - car_point[:,1]) / self.res
            if normalize:
                img_points[:,0] /= self.img_w
                img_points[:,1] /= self.img_h
            return img_points

        cx, cy = car_point
        px = (cx - self.lidar_range['Left']) / self.res;
        py = (self.lidar_range['Front'] - cy) / self.res;
        if normalize:
            px /= self.img_w
            py /= self.img_h
        return np.array([int(px), int(py)], np.int32)
     
    def load_data_to_gpu(self, batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            batch_dict[key] = torch.from_numpy(val).float().cuda()

    def basic_inputs(self, inputs):
        # print('inputs', inputs)
        return {
            # 'traj_ins_all': inputs['traj_ins_all'],
            # 'traj_hmi_all': inputs['traj_hmi_all'],
            'lidar_bev': inputs['lidar_bev'],
            # 'label_bev': inputs['label_bev'],
            'img_hmi': inputs['img_hmi'],
            # 'img_ins': inputs['img_ins'],
            # 'frame_id': inputs['idx'],
            # 'traj_ins': inputs['traj_ins'],
            # 'traj_ins_pixel_norm': inputs['traj_ins_pixel_norm'],
            # 'traj_hmi': inputs['traj_hmi'],
            # 'traj_hist': inputs['traj_hist'],
            # 'heatmap': inputs['heatmap'],
            # 'batch_size': 1,
        }