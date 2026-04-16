#!/usr/bin/env python3
"""
Plot poses from a text file.

Expected input format:
- One pose per line
- 12 floating point values per line
- Interpreted as a 3x4 pose matrix:
      r11 r12 r13 tx
      r21 r22 r23 ty
      r31 r32 r33 tz

This script:
1. Loads poses from a text file
2. Extracts translations
3. Plots the trajectory in 3D
4. Optionally draws local coordinate axes for each pose

Usage:
    python plot_poses.py poses.txt

Optional:
    python plot_poses.py poses.txt --axes
"""

# training data is generated from build_traning_data.py by filtering kitti_raw with trajectory-prediction's labeled frames.

import argparse
import numpy as np
import matplotlib.pyplot as plt


def load_poses(path):
    poses = []
    with open(path, "r") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            vals = [float(x) for x in line.split()]
            if len(vals) != 12:
                raise ValueError(
                    f"Line {line_num}: expected 12 values, got {len(vals)}"
                )

            pose = np.array(vals, dtype=float).reshape(3, 4)
            poses.append(pose)

    if not poses:
        raise ValueError("No poses found in file.")

    return np.array(poses)


def set_axes_equal(ax):
    """
    Make 3D plot axes have equal scale.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    plot_radius = 0.5 * max(x_range, y_range, z_range)

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_poses(poses, draw_axes=False, axis_len=0.3):
    translations = poses[:, :, 3]  # shape: (N, 3)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot trajectory
    ax.plot(
        translations[:, 0],
        translations[:, 1],
        translations[:, 2],
        "-o",
        color="blue",
        markersize=4,
        linewidth=1.5,
        label="trajectory",
    )

    # Mark start and end
    ax.scatter(*translations[0], color="green", s=80, label="start")
    ax.scatter(*translations[-1], color="red", s=80, label="end")

    # Optionally draw pose axes
    if draw_axes:
        for pose in poses:
            R = pose[:, :3]
            t = pose[:, 3]

            x_axis = R[:, 0] * axis_len
            y_axis = R[:, 1] * axis_len
            z_axis = R[:, 2] * axis_len

            ax.quiver(t[0], t[1], t[2], x_axis[0], x_axis[1], x_axis[2],
                      color="r", length=1.0, normalize=False)
            ax.quiver(t[0], t[1], t[2], y_axis[0], y_axis[1], y_axis[2],
                      color="g", length=1.0, normalize=False)
            ax.quiver(t[0], t[1], t[2], z_axis[0], z_axis[1], z_axis[2],
                      color="b", length=1.0, normalize=False)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Pose Trajectory")
    ax.legend()
    set_axes_equal(ax)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("poses_file", help="Path to poses text file")
    parser.add_argument(
        "--axes",
        action="store_true",
        help="Draw local coordinate axes for each pose",
    )
    parser.add_argument(
        "--axis-len",
        type=float,
        default=0.3,
        help="Length of drawn local axes",
    )
    args = parser.parse_args()

    poses = load_poses(args.poses_file)
    plot_poses(poses, draw_axes=args.axes, axis_len=args.axis_len)


if __name__ == "__main__":
    main()
