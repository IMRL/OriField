import numpy as np
import autograd.numpy as anp
from autograd import grad
import torch
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt


def interpolate_xy(A, B, n, mode='right'):
    # Calculate the step size for each dimension
    delta = (B - A) / (n + 1)
    
    # Generate the interpolated points
    interpolated_points = []
    if mode == 'right':
        for i in range(1, n + 1):
            interpolated_points.append(A + i * delta)
    elif mode == 'left':
        for i in range(0, n):
            interpolated_points.append(A + i * delta)
    else:
        raise Exception()
    
    if isinstance(A, np.ndarray):
        return np.stack(interpolated_points, axis=0)
    elif isinstance(A, torch.Tensor):
        return torch.stack(interpolated_points, dim=0)
    else:
        return anp.stack(interpolated_points, axis=0)


def interpolate_map(xy, img_h, img_w):
    if isinstance(xy, np.ndarray):
        idx = xy.copy()
    elif isinstance(xy, torch.Tensor):
        idx = xy.detach().numpy()
    else:
        idx = xy._value.copy()
    idx[:, 0] *= (img_h - 1)
    idx[:, 1] *= (img_w - 1)
    p4 = idx.astype(np.int32)
    p4 = np.where(p4 == 0, p4+1, p4)
    p1 = p4 - 1  # N 2
    p2 = np.stack([p1[:, 0], p4[:, 1]], axis=-1)
    p3 = np.stack([p4[:, 0], p1[:, 1]], axis=-1)

    normalized_p1 = p1.astype(np.float32)
    normalized_p1[:, 0] /= (img_h - 1)
    normalized_p1[:, 1] /= (img_w - 1)
    normalized_p2 = p1.astype(np.float32)
    normalized_p2[:, 0] /= (img_h - 1)
    normalized_p2[:, 1] /= (img_w - 1)
    normalized_p3 = p1.astype(np.float32)
    normalized_p3[:, 0] /= (img_h - 1)
    normalized_p3[:, 1] /= (img_w - 1)
    normalized_p4 = p1.astype(np.float32)
    normalized_p4[:, 0] /= (img_h - 1)
    normalized_p4[:, 1] /= (img_w - 1)
    if isinstance(xy, torch.Tensor):
        normalized_p1 = torch.tensor(normalized_p1)
        normalized_p2 = torch.tensor(normalized_p2)
        normalized_p3 = torch.tensor(normalized_p3)
        normalized_p4 = torch.tensor(normalized_p4)
    d1 = ((xy[:, 0]-normalized_p1[:, 0])**2+(xy[:, 1]-normalized_p1[:, 1])**2)**.5
    d2 = ((xy[:, 0]-normalized_p2[:, 0])**2+(xy[:, 1]-normalized_p2[:, 1])**2)**.5
    d3 = ((xy[:, 0]-normalized_p3[:, 0])**2+(xy[:, 1]-normalized_p3[:, 1])**2)**.5
    d4 = ((xy[:, 0]-normalized_p4[:, 0])**2+(xy[:, 1]-normalized_p4[:, 1])**2)**.5
    w1 = 1-d1/(d1+d2+d3+d4)
    w2 = 1-d2/(d1+d2+d3+d4)
    w3 = 1-d3/(d1+d2+d3+d4)
    w4 = 1-d4/(d1+d2+d3+d4)
    return [p1, p2, p3, p4], [w1, w2, w3, w4]


def interpolate_tangent(ps, ws, tangent_map):  # 4 N 2, 4 N, H W 2
    tangent_map_sum = None
    sum_w = None
    for p, w in zip(ps, ws):
        picked = tangent_map[p.T[0], p.T[1]]*w[:, None]  # N 2
        if tangent_map_sum is None:
            tangent_map_sum = picked
            sum_w = w
        else:
            tangent_map_sum += picked
            sum_w += w
    tangent_map_avg = tangent_map_sum / sum_w[:, None]
    return tangent_map_avg  # N 2


def interpolate_edt(ps, ws, edt_map):
    edt_map_sum = None
    sum_w = None
    for p, w in zip(ps, ws):
        picked = edt_map[p.T[0], p.T[1]]*w[:, None]  # N
        if edt_map_sum is None:
            edt_map_sum = picked
            sum_w = w
        else:
            edt_map_sum += picked
            sum_w += w
    edt_map_avg = edt_map_sum / sum_w[:, None]
    return edt_map_avg  # N


def cal_tengents(starts, ends):  # N 2
    tangents = ends - starts  # N 2
    tangents_length = (tangents[:, 0] ** 2 + tangents[:, 1] ** 2) ** .5  # N
    return tangents / tangents_length[:, None]


def similarity_conv(tangent, tangents):
    dots = tangent*tangents
    similaritys = (dots[:, 0] + dots[:, 1]).mean()
    # print("similaritys", similaritys)
    return similaritys


def blockness_sum(edts):
    blocknesses = edts.mean()
    # print("blocknesses", blocknesses)
    return blocknesses


def path_loss(normalized_path, normalized_path_tengent, tangent_map, edt_map):
    ps, ws = interpolate_map(normalized_path, tangent_map.shape[0], tangent_map.shape[1])
    tangents = interpolate_tangent(ps, ws, tangent_map)
    # edts = interpolate_edt(ps, ws, edt_map)
    # print(tangents, edts)
    step_loss1 = -similarity_conv(normalized_path_tengent, tangents)
    step_loss2 = 0#blockness_sum(edts)
    # print("step_loss1, step_loss2", step_loss1, step_loss2)
    step_loss = step_loss1 + step_loss2
    return step_loss


def traj_loss(normalized_traj, tangent_map, edt_map, implicit_interpolate):
    if implicit_interpolate:
        loss = None
        for i in range(1, len(normalized_traj)):
            interpolates = interpolate_xy(normalized_traj[i-1], normalized_traj[i], 10)
            interpolates_tangents = cal_tengents(normalized_traj[i-1, None], normalized_traj[i, None])
            step_loss = path_loss(interpolates, interpolates_tangents, tangent_map, edt_map)
            if loss is None:
                loss = step_loss
            else:
                loss += step_loss
        loss /= len(normalized_traj) - 1
    else:
        interpolates = normalized_traj[1:]
        interpolates_tangents = cal_tengents(normalized_traj[:-1], normalized_traj[1:])
        loss = path_loss(interpolates, interpolates_tangents, tangent_map, edt_map)
    return loss


traj_grad = grad(traj_loss)


def traversability_to_edt(traversability_map):
    distances, indices = distance_transform_edt(traversability_map, return_indices=True)
    edt_map = 100000000*np.clip(np.exp(-distances/2)*2-1, 0, 1)  # zero point: 1.38
    return edt_map
def traj_grad_wrap_anp(normalized_traj, tangent_map, edt_map, implicit_interpolate):
    normalized_traj = anp.array(normalized_traj)
    loss = np.nan#traj_loss(normalized_traj, tangent_map, edt_map, implicit_interpolate)
    grad = traj_grad(normalized_traj, tangent_map, edt_map, implicit_interpolate)
    return loss, grad
def traj_grad_wrap_torch(normalized_traj, tangent_map, edt_map, implicit_interpolate):
    normalized_traj = torch.tensor(normalized_traj, dtype=torch.float32, requires_grad=True)
    tangent_map = torch.tensor(tangent_map, dtype=torch.float32, requires_grad=False)
    edt_map = torch.tensor(edt_map, dtype=torch.float32, requires_grad=False)
    loss = traj_loss(normalized_traj, tangent_map, edt_map, implicit_interpolate)
    loss.backward()
    grad = normalized_traj.grad
    return loss.detach().numpy(), grad.detach().numpy()
def traj_grad_descent(tangent_map, edt_map, normalized_traj, lr=.01, step=5, implicit_interpolate=True, anp=False, torch=False):
    if not implicit_interpolate:
        new_normalized_traj = []
        for i in range(1, len(normalized_traj)):
            start_end = normalized_traj[i-1:i+1]
            new_normalized_traj.append(interpolate_xy(start_end[0], start_end[1], 10))
        normalized_traj = np.concatenate(new_normalized_traj, axis=0)
    if anp:
        normalized_traj_anp = normalized_traj.copy()
    if torch:
        normalized_traj_torch = normalized_traj.copy()
    for i in range(step):
        # print("step", i, "*"*50)
        if anp:
            loss_anp, grad_anp = traj_grad_wrap_anp(normalized_traj_anp, tangent_map, edt_map, implicit_interpolate)
            # print("normalized_traj_anp", i, normalized_traj_anp)
            # print("loss_anp, grad_anp", loss_anp, grad_anp)
            normalized_traj_anp = np.clip(normalized_traj_anp-grad_anp*lr, 0, 1)
            # print("updated, normalized_traj_anp", normalized_traj_anp)
        if torch:
            loss_torch, grad_torch = traj_grad_wrap_torch(normalized_traj_torch, tangent_map, edt_map, implicit_interpolate)
            # print("normalized_traj_torch", normalized_traj_torch)
            # print("loss_torch, grad_torch", loss_anp, grad_anp)
            normalized_traj_torch = np.clip(normalized_traj_torch-grad_torch*lr, 0, 1)
            # print("updated, normalized_traj_torch", normalized_traj_torch)
        # if anp and torch:
        #     print("loss_diff", loss_anp-loss_torch)
        #     print("grad_diff", (grad_anp-grad_torch).sum())
        #     print("normalized_traj_diff", (normalized_traj_anp-normalized_traj_torch).sum())
    if anp and torch:
        return normalized_traj_anp, normalized_traj_torch
    elif anp:
        return normalized_traj_anp
    elif torch:
        return normalized_traj_torch


if __name__ == '__main__':
    normalized_traj = np.array([
        [0,0],
        [.1,.1],
        [.2,.2],
        [.3,.3],
        [.4,.4],
        [.5,.5],
        [.6,.6],
        [.7,.7],
        [.8,.8],
        [.9,.9],
        [1,1]
    ])
    normalized_traj = np.array([
        [.5,.0],
        [.5,.1],
        [.5,.2],
        [.5,.3],
        [.5,.4],
        [.5,.5],
        [.6,.6],
        [.7,.7],
        [.8,.8],
        [.9,.9],
        [1,1]
    ])
    tangent_map_x = np.array([
        [1.,1,1,1,1],
        [1.,1,1,1,1],
        [1.,1,1,1,1],
        [1.,1,1,1,1],
        [1.,1,1,1,1],
    ])
    tangent_map_y = np.array([
        [0.,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
        [0,0,0,0,0],
    ])
    tangent_map = np.stack([tangent_map_x, tangent_map_y], axis=-1)
    # tangent_map = np.zeros_like(tangent_map)
    traversability_map = np.array([
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 1, 1, 0],
    ])
    # traversability_map = np.ones_like(traversability_map)

    traj_grad_descent(tangent_map, traversability_map, 
                        normalized_traj, anp=True, torch=True)
