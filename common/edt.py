import numpy as np
from scipy.ndimage import distance_transform_edt


def replace_mask_with_nearest(arr, mask=None, distance_limit=None, return_distance=False):
    if mask is None:
        # Create a masked array with mask as the zeros in arr
        masked_arr = np.ma.masked_equal(arr, 0)
        # Compute the distances for each zero entry to the nearest non-zero value
        distances, indices = distance_transform_edt(masked_arr.mask, return_indices=True)
        # Use the indices to map the nearest non-zero values back to the original array's structure
        filled_arr = masked_arr[indices[0], indices[1]]
    else:
        distances, indices = distance_transform_edt(mask, return_indices=True)
        filled_arr = arr[indices[0], indices[1]]
    if distance_limit is not None:
        filled_arr[distances > distance_limit] = arr[distances > distance_limit]
        distances[distances > distance_limit] = 0
    if not return_distance:
        return filled_arr
    else:
        return filled_arr, distances
