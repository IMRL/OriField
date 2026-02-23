import math
import numpy as np
import matplotlib.pyplot as plt

def distance(p1, p2):
    """Calculate the Euclidean distance between two points p1 and p2."""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def extend_path(path, length):
    """Extend the path to reach a specified length from the start to the new endpoint."""
    last_point = path[-1]
    second_last_point = None
    for i in range(len(path)-2, -1, -1):
        direction_length = distance(path[i], last_point)
        if direction_length > 0:
            second_last_point = path[i]
            break

    # Calculate direction vector
    direction = (last_point[0] - second_last_point[0], last_point[1] - second_last_point[1])
    # Normalize direction vector
    direction = (direction[0] / direction_length, direction[1] / direction_length)

    # Calculate the distance to extend
    additional_length = length * 2
    # Calculate the new endpoint
    new_endpoint = (last_point[0] + direction[0] * additional_length, last_point[1] + direction[1] * additional_length)
    return np.concatenate([path, [new_endpoint]])

def find_intersections(path, center, r, num_circles, verbose=False):
    """
    Find intersection points of the path with a fixed number of circles of radii r, 2r, 3r, ...
    
    Parameters:
    path (list of tuples): List of (x, y) coordinates representing the path.
    center (tuple): (x, y) coordinates of the center of the circles.
    r (float): Radius of the first circle.
    num_circles (int): Number of circles.
    
    Returns:
    list of tuples: The intersection points.
    """
    radii = np.linspace(0, r, num=num_circles+1)[1:]

    # Extend the path if necessary
    if len(path) < 2 or distance(path[0], path[-1]) == 0:
        return np.full((radii.shape[0]+1, 2), 0.)
    last_point_distance_to_center = distance(path[-1], center)
    if last_point_distance_to_center < r:
        path = extend_path(path, r)

    p1 = path[:-1]
    p2 = path[1:]

    dx = p2[:, 0] - p1[:, 0]
    dy = p2[:, 1] - p1[:, 1]
    fx, fy = p1[:, 0] - center[0], p1[:, 1] - center[1]

    # Vectorize over radii and segments
    a = dx * dx + dy * dy
    b = 2 * (fx * dx + fy * dy)
    c = (fx * fx + fy * fy) - radii[:, np.newaxis]**2 
    discriminant = b * b - 4 * a * c
    sqrt_discriminant = np.sqrt(discriminant)
    t1 = (-b - sqrt_discriminant) / (2 * a)
    t2 = (-b + sqrt_discriminant) / (2 * a)

    valid_t1 = (0 <= t1) & (t1 <= 1)
    valid_t2 = (0 <= t2) & (t2 <= 1)
    solutions2 = valid_t1 & valid_t2
    solutions1 = valid_t1 | valid_t2
    solutions0 = ~solutions1 
    t1_appro = np.abs(t1 - 0.5)
    t2_appro = np.abs(t2 - 0.5)
    tidx = np.where(solutions2, np.argmax(np.stack([t1, t2], axis=-1), axis=-1), np.argmax(np.stack([valid_t1, valid_t2], axis=-1), axis=-1))  # r seg: t
    tidx = np.where(solutions0, t1_appro > t2_appro, tidx)
    good_solutions = np.argmax(solutions1, axis=-1)
    bad_solutions = np.where(np.min(t1_appro, axis=-1) <= np.min(t2_appro, axis=-1), np.argmin(t1_appro, axis=-1), np.argmin(t2_appro, axis=-1))
    idx = np.where(solutions1.any(-1), good_solutions, bad_solutions)  # r: seg
    St1 = t1[np.arange(radii.shape[0]), idx]
    St2 = t2[np.arange(radii.shape[0]), idx]
    Stidx = tidx[np.arange(radii.shape[0]), idx]
    t12 = np.where(Stidx == 0, St1, St2)
    if verbose:
        print("a.shape, b.shape, c.shape", a.shape, b.shape, c.shape)
        print("discriminant", discriminant)
        print("t1, t2", t1, t2)
        print("solutions1", solutions1)
    if verbose:
        print("t12", t12)
    intersection_x = p1[idx, 0] + t12 * dx[idx]
    intersection_y = p1[idx, 1] + t12 * dy[idx]
    intersection = np.stack((intersection_x, intersection_y), axis=-1)
    if verbose:
        print("intersection", intersection)

    return np.concatenate([np.array([[0.,0.]]), intersection])

def plot_path_with_intersections(ax, path, center, intersections, r, num_circles):
    """
    Plot the path and its intersections with circles of radii r, 2r, 3r, ...

    Args:
    - path (list of lists or numpy array): List of points representing the path.
    - intersections (list of lists): List of intersection points.
    - r (float): Distance interval for the circles.
    """
    radii = np.linspace(0, r, num=num_circles+1)[1:]

    ax.plot(path[:, 0], path[:, 1], '-', label='Path')
    # Plot intersection points
    ax.scatter(intersections[:, 0], intersections[:, 1], color='red', label='Intersections')

    # Plot circles
    for i in radii:
        circle = plt.Circle((center[0], center[1]), i, color='gray', fill=False, linestyle='--')
        ax.add_patch(circle)
    # ax.scatter(center[0], center[1], color='green', label='Start Point')


if __name__ == "__main__":

    while True:
        path = np.array([
            [0, 0],
            [0.1, 1.2],
            [0.25, 1.84],
            [0.36, 2.79 ],
            [0.43, 3.09],
            [0.6, 4.2],
            [0.9, 5.2],
        ])
        path_ref = np.array([
            [0, 0],
            [0, 1.2],
            [0, 1.84],
            [0, 2.79 ],
            [0, 3.09],
            [0, 4.2],
            [0, 5.2],
        ])
        # print("path", path)

        r = 5
        num_circles = 10
        intersections = find_intersections(path, path[0], r, num_circles)
        intersections_ref = find_intersections(path_ref, path_ref[0], r, num_circles)
        plt.figure(figsize=(4, 4))
        plot_path_with_intersections(plt.gca(), path_ref, path_ref[0], intersections_ref, r, num_circles)
        plot_path_with_intersections(plt.gca(), path, path[0], intersections, r, num_circles)
        for inter, inter_ref in zip(intersections, intersections_ref):
            plt.gca().add_line(plt.Line2D((inter[0], inter_ref[0]), (inter[1], inter_ref[1]), color='red', linestyle='--'))
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.legend()
        # plt.grid()
        plt.title('Path and Intersections with Circles')
        plt.axis('equal')
        plt.show()

        path = np.random.rand(5, 2) * np.linspace(0, 5, num=5)[:, None]

        r = 5
        num_circles = 10
        intersections = find_intersections(path, path[0], r, num_circles)
        plt.figure(figsize=(4, 4))
        plot_path_with_intersections(plt.gca(), path, path[0], intersections, r, num_circles)
        plt.xlabel('X')
        plt.ylabel('Y')
        # plt.legend()
        # plt.grid()
        plt.title('Path and Intersections with Circles')
        plt.axis('equal')
        plt.show()