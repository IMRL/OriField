import numpy as np


# Function to calculate a point on a quadratic Bézier curve
def bezier_point(t, p0, p1, p2):
    """Calculate the quadratic Bézier curve point and tangent at parameter t."""
    # Calculate the point
    point = (
        (1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * p1[0] + t ** 2 * p2[0],
        (1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * p1[1] + t ** 2 * p2[1]
    )

    # Calculate the tangent
    tangent = (
        2 * (1 - t) * (p1[0] - p0[0]) + 2 * t * (p2[0] - p1[0]),
        2 * (1 - t) * (p1[1] - p0[1]) + 2 * t * (p2[1] - p1[1])
    )

    # Optionally, normalize the tangent vector
    tangent_length = (tangent[0]**2 + tangent[1]**2) ** 0.5
    if tangent_length != 0:
        tangent = (tangent[0] / tangent_length, tangent[1] / tangent_length)

    return point, tangent


# Function to draw a quadratic Bézier curve on an image
def bezier_curve(P0, P1, P2):
    num_steps = 100
    points = []
    tangents = []
    
    # Generate points along the curve
    for i in range(num_steps + 1):
        t = i / num_steps
        point, tangent = bezier_point(t, P0, P1, P2)
        points.append(point)
        tangents.append(tangent)

    return points, tangents


def catmull_rom_spline(t, a, b, c, d):
    """Calculate the Catmull-Rom spline point and tangent at parameter t."""
    # Catmull-Rom spline matrix
    M = np.array([
        [-0.5, 1.5, -1.5, 0.5],
        [1.0, -2.5, 2.0, -0.5],
        [-0.5, 0.0, 0.5, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ])
    
    # Create a matrix of points
    P = np.array([a, b, c, d])
    
    # Calculate the point
    T = np.array([t**3, t**2, t, 1])
    point = T @ M @ P

    # Calculate the tangent
    Td = np.array([3 * t**2, 2 * t, 1, 0])
    tangent = Td @ M @ P

    # Normalize the tangent
    tangent_length = np.linalg.norm(tangent)
    if tangent_length != 0:
        tangent = tangent / tangent_length

    return point, tangent


def catmull_rom_point(t, p0, p1, p2):
    # Virtual points
    p_3 = 2 * p0 - p1  # Simple estimation
    p3 = 2 * p2 - p1   # Simple estimation

    if t <= .5:
        return catmull_rom_spline(t * 2, p_3, p0, p1, p2)
    else:
        return catmull_rom_spline((t - .5) * 2, p0, p1, p2, p3)


# Function to draw a catmull rom curve on an image
def catmull_rom_curve(P0, P1, P2):
    num_steps = 100
    points = []
    tangents = []
    
    # Generate points along the curve
    for i in range(num_steps + 1):
        t = i / num_steps
        point, tangent = catmull_rom_point(t, P0, P1, P2)
        points.append(point)
        tangents.append(tangent)

    return points, tangents


def curve(start_point, control_point, end_point):
    return bezier_curve(start_point, control_point, end_point)
    # return catmull_rom_curve(start_point, control_point, end_point)


def curve_by_curviness(start_point, end_point, curviness):
    midpoint = ((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
    control_point = (
        midpoint[0] + curviness * (end_point[1] - start_point[1]) / 2,
        midpoint[1] - curviness * (end_point[0] - start_point[0]) / 2
    )
    return curve(start_point, control_point, end_point)


def curve_by_control(start_point, end_point, control_point):
    return curve(start_point, control_point, end_point)
