import numpy as np
from scipy.ndimage import distance_transform_edt
import cv2
from skimage import data
import matplotlib.pyplot as plt
from . import planning


def edt_with_tangent(mask, boundary, invalid=None):
    inverse_mask = ~mask
    if invalid is None:
        invalid = np.zeros_like(mask)
        
    # Compute the Euclidean distance transform
    distances, indices = distance_transform_edt(mask | invalid, return_indices=True)
    negative_distances = -distances
    inverse_distances, inverse_indices = distance_transform_edt(inverse_mask | invalid, return_indices=True)
    negative_inverse_distances = -inverse_distances
    distances2 = (distances + negative_inverse_distances) * mask
    inverse_distances2 = (inverse_distances + negative_distances) * inverse_mask

    # Apply Sobel filters to get gradients in x and y directions
    grad_x = cv2.Sobel(distances2, cv2.CV_64F, 0, 1, ksize=5) * mask
    grad_y = cv2.Sobel(distances2, cv2.CV_64F, 1, 0, ksize=5) * mask
    # Calculate gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # Normalize gradients
    grad_x = np.where(grad_magnitude != 0, grad_x / grad_magnitude, grad_x)
    grad_y = np.where(grad_magnitude != 0, grad_y / grad_magnitude, grad_y)
    tangent_x = -grad_y
    tangent_y = grad_x.copy()
    # grad_x_in_key = np.full(grad_x.shape, 0.)
    # grad_y_in_key = np.full(grad_y.shape, 0.)
    # grad_x_in_key[key] = grad_x[key]
    # grad_y_in_key[key] = grad_y[key]
    # grad_magnitude_in_key = np.sqrt(grad_x_in_key**2 + grad_y_in_key**2)
    # tangent_x_in_key = -grad_y_in_key
    # tangent_y_in_key = grad_x_in_key.copy()
    dij = planning.Dijkstra(mask)
    movements, tangents = dij.plan(boundary)
    dij_tangent_x = tangents[..., 0]
    dij_tangent_y = tangents[..., 1]
    dij_tangent_magnitude = np.sqrt(dij_tangent_x**2 + dij_tangent_y**2)
    d_tangent_sign = np.where(tangent_x * dij_tangent_x + tangent_y * dij_tangent_y >= 0, 1, -1)
    d_tangent_x = tangent_x * d_tangent_sign
    d_tangent_y = tangent_y * d_tangent_sign
    d_tangent_magnitude = np.sqrt(d_tangent_x**2 + d_tangent_y**2)

    inverse_grad_x = cv2.Sobel(inverse_distances2, cv2.CV_64F, 0, 1, ksize=5) * inverse_mask
    inverse_grad_y = cv2.Sobel(inverse_distances2, cv2.CV_64F, 1, 0, ksize=5) * inverse_mask
    inverse_grad_magnitude = np.sqrt(inverse_grad_x**2 + inverse_grad_y**2)
    inverse_grad_x = np.where(inverse_grad_magnitude != 0, inverse_grad_x / inverse_grad_magnitude, inverse_grad_x)
    inverse_grad_y = np.where(inverse_grad_magnitude != 0, inverse_grad_y / inverse_grad_magnitude, inverse_grad_y)

    dists = distances2.copy()
    grads = np.stack([grad_x, grad_y], axis=-1)
    tangents = np.stack([tangent_x, tangent_y], axis=-1)
    dij_tangents = np.stack([dij_tangent_x, dij_tangent_y], axis=-1)
    d_tangents = np.stack([d_tangent_x, d_tangent_y], axis=-1)
    inverse_dists = inverse_distances2.copy()
    inverse_grads = np.stack([inverse_grad_x, inverse_grad_y], axis=-1)
    combines_dists = np.where(mask, dists, inverse_dists)
    combines = np.where(mask[..., None], d_tangents, inverse_grads)

    # def tengent_to_color(tengent, tengent_magnitude):
    #     color_tengent = plt.get_cmap('hsv')((np.arctan2(tengent[..., 1], tengent[..., 0]) / np.pi + 1) / 2)
    #     color_tengent[..., -1] = tengent_magnitude
    #     return color_tengent
    # fig, axes = plt.subplots(8, 2, figsize=(8, 8), sharex=True, sharey=True)
    # ax = axes.ravel()
    # ax[0].imshow(mask)
    # ax[1].imshow(inverse_mask)
    # ax[2].imshow(distances)
    # ax[3].imshow(inverse_distances)
    # ax[4].imshow(negative_distances)
    # ax[5].imshow(negative_inverse_distances)
    # ax[6].imshow(distances2)
    # ax[7].imshow(inverse_distances2)

    # ax[8].imshow(tengent_to_color(grads, grad_magnitude))
    # ax[9].imshow(tengent_to_color(tangents, grad_magnitude))
    # ax[10].imshow(tengent_to_color(dij_tangents, dij_tangent_magnitude))
    # ax[11].imshow(tengent_to_color(d_tangents, d_tangent_magnitude))

    # ax[12].imshow(tengent_to_color(inverse_grads, inverse_grad_magnitude))
    # ax[13].imshow(tengent_to_color(combines, np.full(combines.shape[:-1], 1.)))

    # ax[14].imshow(invalid)

    # import matplotlib as mpl
    # cmap = plt.cm.get_cmap('hsv')
    # display_axes = fig.add_axes([0.95, 0.95, 0.05, 0.05], projection='polar')
    # display_axes.set_theta_offset(np.pi/2)
    # bar = mpl.colorbar.ColorbarBase(display_axes, cmap=cmap,
    #                         norm=mpl.colors.Normalize(0.0, 2*np.pi),
    #                         orientation='horizontal')
    # bar.outline.set_visible(False)
    # plt.show()

    return dists, grads, tangents, dij_tangents, d_tangents, inverse_dists, inverse_grads, combines_dists, combines


if __name__ == "__main__":
    mask = ~data.horse()
    boundary = np.zeros_like(mask)
    boundary[0:20, :] = True
    mask[boundary] = True
    invalid = np.zeros_like(mask)
    invalid[-50:, :] = True
    edt_with_tangent(mask, boundary, invalid=invalid)