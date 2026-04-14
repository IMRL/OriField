import os
import numpy as np
from datetime import datetime
import pykitti
from tqdm import tqdm


def find_closest_indices(list_A, list_B):
    """
    Find the index in list B for each item in list A by the closest one.

    Parameters:
    list_A (list of datetime): List of datetime objects to find closest matches for.
    list_B (list of datetime): List of datetime objects to search in.

    Returns:
    list of int: List of indices in list B corresponding to the closest datetime in list B for each datetime in list A.
    """
    closest_indices = []
    
    for a in list_A:
        closest_index = min(range(len(list_B)), key=lambda i: abs((list_B[i] - a)))
        closest_indices.append(closest_index)
    
    return closest_indices


seqs_drives_mapping = {
    "00": "2011_10_03_drive_0027",
    "01": "2011_10_03_drive_0042",
    "02": "2011_10_03_drive_0034",
    "03": "2011_09_26_drive_0067",
    "04": "2011_09_30_drive_0016",
    "05": "2011_09_30_drive_0018",
    "06": "2011_09_30_drive_0020",
    "07": "2011_09_30_drive_0027",
    "08": "2011_09_30_drive_0028",
    "09": "2011_09_30_drive_0033",
    "10": "2011_09_30_drive_0034",
    "11": "2011_09_30_drive_0035",
    "12": "2011_09_30_drive_0036",
    "13": "2011_09_30_drive_0037",
    "14": "2011_09_30_drive_0038",
    "15": "2011_09_30_drive_0039",
    "16": "2011_09_30_drive_0040",
    "17": "2011_09_30_drive_0041",
    "18": "2011_09_30_drive_0042",
    "19": "2011_09_30_drive_0043",
    "20": "2011_09_30_drive_0044",
    "21": "2011_09_30_drive_0045"
}
seqs = ["00", "02", "05", "07", "08", "09", "10"]

for seq in seqs:
    hmi_file = f"./kitti/kitti_{seq}_hmi.csv"
    ins_file = f"./kitti/kitti_{seq}.csv"
    with open(hmi_file, "r") as f:
        hmi_times = []
        hmi_llu = []
        lines = f.readlines()
        for line in lines:
            l = line.strip().split(',')
            timestamp = datetime.fromtimestamp(float(l[0]))  # in the formart of 0.0
            hmi_times.append(timestamp)
            hmi_llu.append(np.fromstring(','.join(l[1:4]), dtype=float, sep=','))
        hmi_times = np.array(hmi_times)
        hmi_llu = np.array(hmi_llu)
        # easting_northing = utm.from_latlon(hmi_llu[:, 0], hmi_llu[:, 1])
        # easting, northing = easting_northing[0], easting_northing[1]
        # hmi_en = np.stack([easting, northing], axis=-1)
    with open(ins_file, "r") as f:
        ins_times = []
        ins_llu = []
        lines = f.readlines()
        for line in lines:
            l = line.strip().split(',')
            timestamp = datetime.fromtimestamp(float(l[0][:-3]))  # in the formart of 1317359776.230110883
            ins_times.append(timestamp)
            ins_llu.append(np.fromstring(','.join(l[1:4]), dtype=float, sep=','))
        ins_times = np.array(ins_times)
        ins_llu = np.array(ins_llu)
        # easting_northing = utm.from_latlon(ins_llu[:, 0], ins_llu[:, 1])
        # easting, northing = easting_northing[0], easting_northing[1]
        # ins_en = np.stack([easting, northing], axis=-1)
    date, drive = seqs_drives_mapping[seq].split('_drive_')
    data = pykitti.raw("./kitti-raw", date, drive)
    scale = np.cos(data.oxts[0].packet.lat * np.pi / 180.)
    origin = pykitti.utils.pose_from_oxts_packet(data.oxts[0].packet, scale)[1]
    hmi_enu = []
    for i in range(len(hmi_llu)):
        class Object(object):
            pass
        packet = Object()
        packet.lat = hmi_llu[i, 0]
        packet.lon = hmi_llu[i, 1]
        packet.alt = 0  # hmi_llu[i, 2]
        packet.roll = 0
        packet.pitch = 0
        packet.yaw = 0
        R, t = pykitti.utils.pose_from_oxts_packet(packet, scale)
        hmi_enu.append(t-origin)
    hmi_enu = np.array(hmi_enu)
    ins_enu = []
    for i in range(len(ins_llu)):
        class Object(object):
            pass
        packet = Object()
        packet.lat = ins_llu[i, 0]
        packet.lon = ins_llu[i, 1]
        packet.alt = 0  # ins_llu[i, 2]
        packet.roll = 0
        packet.pitch = 0
        packet.yaw = 0
        R, t = pykitti.utils.pose_from_oxts_packet(packet, scale)
        ins_enu.append(t)
    ins_enu = np.array(ins_enu-origin)
    data_enu = []
    for i in range(len(data.oxts)):
        packet = data.oxts[i].packet
        R, t = pykitti.utils.pose_from_oxts_packet(packet, scale)
        data_enu.append(t-origin)
    data_enu = np.array(data_enu)

    key_frames = find_closest_indices(ins_times, data.timestamps)
    poses = []
    for i in range(len(key_frames)):
        key_frame = key_frames[i]
        pose = np.dot(data.oxts[key_frame].T_w_imu, np.linalg.inv(data.calib.T_velo_imu))
        poses.append(pose)
    poses = np.array(poses)
    save_dir = f"./raw-derived-semantic-kitti/sequences/{seq}/velodyne"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, "poses.txt"), "w") as f:
        f.write('\n'.join([' '.join(list(map(str, item[:3].reshape((-1)).tolist()))) for item in poses]))
    for i in tqdm(range(len(poses))):
        key_frame = key_frames[i]

        velo = data.get_velo(key_frame)
        velo.reshape((-1)).tofile(os.path.join(save_dir, f"{i:06d}.bin"))
        velo = np.stack([-velo[:, 1], velo[:, 0], velo[:, 2], velo[:, 3]], axis=-1)