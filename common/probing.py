import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
import prettypyplot as pplt
from common.visulization import Visulizater


def norm(x, stat=None):
    if stat is None:
        mean, std = np.mean(x), np.std(x)
    else:
        mean, std = np.mean(stat), np.std(stat)
    return (x - mean) / std


def feature_dist(x, nonzero_channels=[]):  # B, C, H, W
    B, C, H, W = x.shape
    channels = []
    for c in range(C):
        channels.append(x[:, c])
    fig, axes = plt.subplots(4, C, figsize=(10, 10))#, sharex=True, sharey=True)
    # pplt.use_style()
    axes_origin, axes_batch_norm_batch, axes_instance_norm, axes_instance_norm_batch = axes
    for c in range(C):
        l_bn = []
        l_cn = []
        batch_features = channels[c].reshape(-1).cpu().detach().numpy()
        if c in nonzero_channels:
            bx = batch_features[batch_features != 0]
        else:
            bx = batch_features.copy()
        for b in range(B):
            features = channels[c][b].reshape(-1).cpu().detach().numpy()
            if c in nonzero_channels:
                cx = features[features != 0]
            else:
                cx = features.copy()
            sns.kdeplot(cx, ax=axes_origin[c] if isinstance(axes_origin, np.ndarray) else axes_origin)
            bn = norm(cx, stat=bx)
            cn = norm(cx)
            l_bn.append(bn)
            l_cn.append(cn)
            # sns.kdeplot(bn, ax=axes_batch_norm[c] if isinstance(axes_batch_norm, np.ndarray) else axes_batch_norm)
            sns.kdeplot(cn, ax=axes_instance_norm[c] if isinstance(axes_instance_norm, np.ndarray) else axes_instance_norm)
        sns.kdeplot(np.concatenate(l_bn), ax=axes_batch_norm_batch[c] if isinstance(axes_batch_norm_batch, np.ndarray) else axes_batch_norm_batch)
        sns.kdeplot(np.concatenate(l_cn), ax=axes_instance_norm_batch[c] if isinstance(axes_instance_norm_batch, np.ndarray) else axes_instance_norm_batch)
    ax = axes.ravel()
    for a in ax:
        a.set_xlabel("")  # Remove x-axis name
        a.set_ylabel("")  # Remove y-axis name
    plt.show()
    # for a in dummy:
    #     a.axis("off")
    # for a in dummy2:
    #     a.axis("off")
    # for a in dummy3:
    #     a.axis("off")
    # for a in dummy4:
    #     a.axis("off")
    fig, axes = plt.subplots(B//2, 2, figsize=(10, 10))#, sharex=True, sharey=True)
    for b in range(B):
        bev_map = x[b, :3].permute(1, 2, 0).cpu().detach().numpy()
        tangent_init = x[b, 3:5].permute(1, 2, 0).cpu().detach().numpy()
        visulizater = Visulizater(None)
        axesimg = fig.add_subplot(B//2, 2, 1+b)
        axesimg.axis("off")
        axesimg.imshow(np.concatenate([
            visulizater.naive_vis(bev_map), 
            visulizater.tangent_vis(tangent_init, pltcm='hsv')[..., :3]
        ], axis=1))
    ax = axes.ravel()
    for a in ax:
        a.axis("off")
    # plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()