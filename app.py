import glob
import os
import time
import argparse
import threading

import gradio as gr
import numpy as np
from scipy.spatial import KDTree
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.modeling import build_model

from lidardet.ops.lidar_bev import bev

from model import setup as setup_cfg
from common.peroidic import peroidic_np
from common.polar import *

PATH = os.path.dirname(os.path.abspath(__file__))


def get_pred(tangent_init_polar, pred):
    return tangent_init_polar + pred


# ── Visualizer ────────────────────────────────────────────────────────────────
class Visulizater:
    def __init__(self):
        pass

    def tangent_vis(self, tangent_map, tangent_mask=None, pltcm="hsv"):
        tangent_map_length = (tangent_map[..., 0]**2 + tangent_map[..., 1]**2) ** .5
        offseted_map_polar = np.arctan2(tangent_map[..., 1], tangent_map[..., 0])
        tangent_vis = self.colormap_vis((offseted_map_polar / np.pi + 1) / 2, pltcm)
        tangent_vis[..., -1] = tangent_map_length
        if tangent_mask is not None:
            tangent_vis[~tangent_mask] = np.array([0, 0, 0, 0.])
        return tangent_vis

    def repeat_vis(self, value):
        return np.repeat(value[..., None], 3, axis=-1)

    def naive_vis(self, value):
        return value

    def colormap_vis(self, value, pltcm):
        return plt.get_cmap(pltcm)(value)


# ── Geometry helpers ──────────────────────────────────────────────────────────
def pierce_tangent(points, tangents, shape):
    tangent_result  = np.full(shape + (tangents.shape[-1],), 0.)
    distance_result = np.full(shape, 0.)
    if points.shape[0] > 0 and tangents.shape[0] > 0:
        tangents_mask_points = np.mgrid[:shape[0], :shape[1]].reshape(2, -1).T
        kdtree = KDTree(points)
        distances, indices = kdtree.query(tangents_mask_points)
        tangent_result [tangents_mask_points.T[0], tangents_mask_points.T[1]] = tangents[indices]
        distance_result[tangents_mask_points.T[0], tangents_mask_points.T[1]] = distances
    return tangent_result, distance_result


def bezier_point(t, p0, p1, p2):
    point = (
        (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0],
        (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1],
    )
    tangent = (
        2*(1-t) * (p1[0]-p0[0]) + 2*t * (p2[0]-p1[0]),
        2*(1-t) * (p1[1]-p0[1]) + 2*t * (p2[1]-p1[1]),
    )
    length = (tangent[0]**2 + tangent[1]**2) ** 0.5
    if length != 0:
        tangent = (tangent[0]/length, tangent[1]/length)
    return point, tangent


def bezier_curve(P0, P1, P2):
    num_steps = 100
    points, tangents = [], []
    for i in range(num_steps + 1):
        t = i / num_steps
        p, tg = bezier_point(t, P0, P1, P2)
        points.append(p)
        tangents.append(tg)
    return points, tangents


def curve(start_point, control_point, end_point):
    return bezier_curve(start_point, control_point, end_point)


def curve_by_control(start_point, end_point, control_point):
    return curve(start_point, control_point, end_point)


def traj_to_points_tangents(traj_points, traj_control):
    traj_tangents = traj_points[1:] - traj_points[:-1]
    traj_tangents_length = (traj_tangents[..., 0]**2 + traj_tangents[..., 1]**2) ** 0.5
    traj_tangents = np.where(
        traj_tangents_length[..., None] != 0,
        traj_tangents / traj_tangents_length[..., None],
        traj_tangents,
    )

    curve_points, curve_tangents = curve_by_control(traj_points[0], traj_points[-1], traj_control)
    curve_points   = np.array(curve_points,   dtype=np.int32)
    curve_tangents = np.array(curve_tangents, dtype=np.float32)
    mask = np.linalg.norm(curve_tangents, axis=-1) != 0
    curve_points   = curve_points[mask]
    curve_tangents = curve_tangents[mask]
    traj_points    = traj_points[:-1]

    return traj_points, traj_tangents, curve_points, curve_tangents


def points_tangents_to_map(traj_points, traj_tangents, curve_points, curve_tangents, shape):
    traj_tangents_map,  traj_distances_map  = pierce_tangent(traj_points,  traj_tangents,  shape)
    curve_tangents_map, curve_distances_map = pierce_tangent(curve_points, curve_tangents, shape)

    traj_mask  = ((traj_points[:,0]  >= 0) & (traj_points[:,0]  < img_h) &
                  (traj_points[:,1]  >= 0) & (traj_points[:,1]  < img_w))
    curve_mask = ((curve_points[:,0] >= 0) & (curve_points[:,0] < img_h) &
                  (curve_points[:,1] >= 0) & (curve_points[:,1] < img_w))

    traj_points_map  = np.full((img_h, img_w), False)
    curve_points_map = np.full((img_h, img_w), False)
    traj_points_map [traj_points [traj_mask,  0], traj_points [traj_mask,  1]] = True
    curve_points_map[curve_points[curve_mask, 0], curve_points[curve_mask, 1]] = True

    return (traj_points_map,  traj_tangents_map,  traj_distances_map,
            curve_points_map, curve_tangents_map, curve_distances_map)


def convert_traj(traj_points, traj_control, shape):
    traj_points, traj_tangents, curve_points, curve_tangents = traj_to_points_tangents(
        traj_points, traj_control
    )
    (traj_points_map, traj_tangents_map, traj_distances_map,
     curve_points_map, curve_tangents_map, curve_distances_map) = points_tangents_to_map(
        traj_points, traj_tangents, curve_points, curve_tangents, shape
    )
    traj_tangents,  traj_tangents_map  = -traj_tangents,  -traj_tangents_map
    curve_tangents, curve_tangents_map = -curve_tangents, -curve_tangents_map

    return (traj_points_map, traj_tangents, traj_tangents_map, traj_distances_map,
            curve_points, curve_points_map, curve_tangents, curve_tangents_map, curve_distances_map)


def perpendicular_control(points):
    a, b = points[0], points[-1]
    ab = (b - a)[None]
    ap = points - a[None]
    if (a != b).sum() == 0:
        length = np.linalg.norm(ap, axis=-1)
    else:
        proj   = ((ap * ab).sum(-1, keepdims=True) / (ab * ab).sum(-1, keepdims=True)) * ab
        length = np.linalg.norm(ap - proj, axis=-1)
    return points[np.argmax(length)]


def normalize_distance(distances, img_h, img_w):
    return 1 - distances / (min(img_h, img_w) / 2)


# ── File helpers ──────────────────────────────────────────────────────────────
def find_example_subdirs(base_folder):
    if not os.path.exists(base_folder):
        return []
    subdirs = sorted(
        item for item in os.listdir(base_folder)
        if os.path.isdir(os.path.join(base_folder, item))
    )
    return subdirs


def find_npy_files_in_subdir(base_folder, subdir=None):
    if subdir is None:
        return []
    folder_path = os.path.join(base_folder, subdir)
    if not os.path.exists(folder_path):
        return []
    npy_files = glob.glob(os.path.join(folder_path, "*.npy"))
    numeric_files = [
        f for f in npy_files
        if os.path.basename(f).replace('.npy','').isdigit()
        and len(os.path.basename(f).replace('.npy','')) == 6
    ]
    numeric_files.sort(key=lambda x: int(os.path.basename(x).replace('.npy', '')))
    return numeric_files


def load_image_and_npy(file_path, type):
    if file_path is None:
        return None, None
    npy_data = np.load(file_path)
    if type == 'bev':
        img = Image.fromarray((np.clip(visulizater.naive_vis(npy_data), 0, 1) * 255).astype(np.uint8))
    elif type == 'tangent':
        img = Image.fromarray((visulizater.tangent_vis(npy_data, pltcm='hsv') * 255).astype(np.uint8))
    elif type == 'distance':
        img = Image.fromarray((np.clip(visulizater.repeat_vis(npy_data), 0, 1) * 255).astype(np.uint8))
    return img, npy_data


def draw_points_on_image(image, points):
    if image is None:
        return None
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)
    for i, (x, y) in enumerate(points):
        draw.ellipse([x-3, y-3, x+3, y+3], fill='red', outline='red')
        draw.text((x+5, y-5), str(i+1), fill='blue')
    if len(points) > 1:
        for i in range(len(points)-1):
            draw.line([tuple(points[i]), tuple(points[i+1])], fill='green', width=2)
    return img_copy


def process_trajectory(points, image_shape):
    if len(points) < 3:
        return None, None, None, None, "Need at least 3 points"
    points_array  = np.array(points)
    traj_control  = perpendicular_control(points_array)
    (_, _, _, _, _, _, curve_tangents, curve_tangents_map, curve_distances_map) = convert_traj(
        points_array, traj_control, image_shape
    )
    tangent_init  = curve_tangents_map.copy()
    distance_init = normalize_distance(curve_distances_map, img_h=img_h, img_w=img_w)

    tangent_vis_img  = (visulizater.tangent_vis(tangent_init, pltcm='hsv') * 255).astype(np.uint8)
    distance_vis_img = (np.clip(visulizater.repeat_vis(distance_init), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(tangent_vis_img), Image.fromarray(distance_vis_img), tangent_init, distance_init, "Success"


# ── Globals ───────────────────────────────────────────────────────────────────
img_h = 400
img_w = 400
visulizater = Visulizater()

example_dir            = "/home/yuminghuang/dataset/basic-bev-kitti/bev_map"
available_subdirs      = []
current_available_files = []

device      = "cuda"
torch_dtype = torch.float32
model       = None
model_status_message = "Initializing..."
model_loading        = False
initial_status = ""


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model(cfg):
    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.to(torch_dtype).to(device)
    model.eval()
    return model


def find_default_model():
    for p in ["model.ckpt", "checkpoints/model.ckpt",
              "models/latest.pth", "weights/orientation_field.pt", "best_model.ckpt"]:
        if os.path.exists(p):
            return p
    return None


def load_model_async(cfg: str, ui_update_callback=None):
    """Load model asynchronously in a separate thread"""
    global model, model_status_message, model_loading

    def _load():
        global model, model_status_message, model_loading
        try:
            model_loading = True
            model_status_message = f"🔄 Loading model from: {cfg.MODEL.WEIGHTS}..."
            print(model_status_message)
            model = load_model(cfg)
            model_status_message = f"✅ Model loaded successfully from: {cfg.MODEL.WEIGHTS}"
            print("Model loaded successfully!")
            if ui_update_callback:
                ui_update_callback()
        except Exception as e:
            model_status_message = f"❌ Error loading model: {e}"
            model = None
            print(model_status_message)
            if ui_update_callback:
                ui_update_callback()
        finally:
            model_loading = False

    # _load()
    # return

    thread = threading.Thread(target=_load, daemon=True)
    thread.start()
    return thread


def parse_args():
    parser = argparse.ArgumentParser(description="OriField Gradio Demo")
    parser.add_argument("--port",       type=int, default=7860)
    parser.add_argument("--share",      action="store_true")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args()


# ── Startup ───────────────────────────────────────────────────────────────────
args = parse_args()
cfg = setup_cfg(args)

available_subdirs = find_example_subdirs(example_dir)
if available_subdirs:
    current_available_files = find_npy_files_in_subdir(example_dir, available_subdirs[0])
print(f"Found {len(available_subdirs)} subdirectories: {available_subdirs}")
if current_available_files:
    print(f"Found {len(current_available_files)} .npy files in first subdirectory")

model_path = None
if cfg.MODEL.WEIGHTS:
    model_path = cfg.MODEL.WEIGHTS
if model_path is None:
    model_path = find_default_model()

if model_path and os.path.exists(model_path):
    initial_status = f"🔄 Will load model from: {model_path}..."
    print(f"Will load model from: {model_path}")
else:
    initial_status = "⚠️ No model specified or found. Please upload a model checkpoint."
    model_path = None
    print("No model path specified or found")

# Start async model loading if path is available
if model_path:
    load_model_async(cfg, lambda: print("Initial model load completed"))


# ── Data classes ──────────────────────────────────────────────────────────────
class Environment:
    def __init__(self):
        self.img1 = None
        self.npy1 = None


class TrajectoryGroup:
    def __init__(self, group_id):
        self.group_id = group_id
        self.points   = []
        self.img2     = None
        self.npy2     = None   # tangent map
        self.img3     = None
        self.npy3     = None   # distance map


def create_trajectory_group(group_id, env_data_state, groups_data_state):
    """Create UI components for a trajectory group"""
    
    with gr.Group():
        gr.Markdown(f"### Group {group_id + 1}")
        
        with gr.Row():
            with gr.Column(scale=1):
                img2_display = gr.Image(label="Image 2 (Tangents)", type="pil")
                img2_upload = gr.File(label="Upload Image 2 (PNG)", file_types=[".npy"], height=60)
                
            with gr.Column(scale=1):
                img3_display = gr.Image(label="Image 3 (Distances)", type="pil")
                img3_upload = gr.File(label="Upload Image 3 (PNG)", file_types=[".npy"], height=60)
        
            with gr.Column(scale=1):
                # interactive_img = gr.Image(
                #     label="Click to add trajectory points", 
                #     type="pil",
                #     interactive=True,
                # )
                interactive_img = gr.ImageEditor(
                    label="Click to add trajectory points", 
                    type="pil",
                )
                points_list = gr.Textbox(label="Points", value="[]", interactive=False)
                clear_btn = gr.Button("Clear Points")
                process_btn = gr.Button("Process Trajectory")
                status = gr.Textbox(label="Status", value="Ready", interactive=False)
        
        # Handle file uploads
        def upload_img2(file, env_data, groups_data):
            if file:
                img, npy = load_image_and_npy(file.name, 'tangent')
                groups_data[group_id].img2 = img
                groups_data[group_id].npy2 = npy
                return img, env_data, groups_data
            return None, env_data, groups_data
        
        def upload_img3(file, env_data, groups_data):
            if file:
                img, npy = load_image_and_npy(file.name, 'distance')
                groups_data[group_id].img3 = img
                groups_data[group_id].npy3 = npy
                return img, env_data, groups_data
            return None, env_data, groups_data
        
        # Handle point clicking
        def add_point(evt: gr.SelectData, env_data, groups_data):
            if env_data.img1 is None:
                return None, "[]", "Please upload Image 1 first", env_data, groups_data
            
            groups_data[group_id].points.append([evt.index[0], evt.index[1]])
            img_with_points = draw_points_on_image(env_data.img1, groups_data[group_id].points)
            return img_with_points, str(groups_data[group_id].points), f"Added point {len(groups_data[group_id].points)}", env_data, groups_data
        
        def clear_points(env_data, groups_data):
            groups_data[group_id].points = []
            return env_data.img1, "[]", "Points cleared", env_data, groups_data
        
        def generate_trajectory(env_data, groups_data):
            if env_data.img1 is None:
                return None, None, "Please upload Image 1 first", env_data, groups_data
            
            if len(groups_data[group_id].points) < 3:
                return None, None, "Need at least 3 points", env_data, groups_data
            
            tangent_img, distance_img, tangent_init, diatance_init, status_msg = process_trajectory(
                np.array(groups_data[group_id].points)[:, ::-1],
                image_shape=(env_data.img1.height, env_data.img1.width)
            )
            
            if tangent_img and distance_img:
                groups_data[group_id].img2 = tangent_img
                groups_data[group_id].img3 = distance_img
                groups_data[group_id].npy2 = tangent_init
                groups_data[group_id].npy3 = diatance_init
                return tangent_img, distance_img, "Trajectory processed successfully", env_data, groups_data
            
            return None, None, status_msg, env_data, groups_data
        
        # Set up initial interactive image
        def setup_interactive(env_data, groups_data):
            if env_data.img1:
                return env_data.img1, env_data, groups_data
            return None, env_data, groups_data
        
        # Connect events
        img2_upload.change(upload_img2, inputs=[img2_upload, env_data_state, groups_data_state], outputs=[img2_display, env_data_state, groups_data_state])
        img3_upload.change(upload_img3, inputs=[img3_upload, env_data_state, groups_data_state], outputs=[img3_display, env_data_state, groups_data_state])
        
        interactive_img.select(add_point, inputs=[env_data_state, groups_data_state], outputs=[interactive_img, points_list, status, env_data_state, groups_data_state])
        clear_btn.click(clear_points, inputs=[env_data_state, groups_data_state], outputs=[interactive_img, points_list, status, env_data_state, groups_data_state])
        process_btn.click(generate_trajectory, inputs=[env_data_state, groups_data_state], outputs=[img2_display, img3_display, status, env_data_state, groups_data_state])
        
        return img2_display, img3_display, interactive_img, setup_interactive


# ── Selection helper ──────────────────────────────────────────────────────────
def load_npy_from_selection(subdir_index, file_index, available_subdirs, env_data, groups_data):
    """Load npy file based on subdirectory and file selection"""
    if not available_subdirs or subdir_index >= len(available_subdirs):
        return None, "[]", "Invalid subdirectory selection", env_data, groups_data
    
    selected_subdir = available_subdirs[subdir_index]
    available_files = find_npy_files_in_subdir(example_dir, selected_subdir)
    
    if not available_files or file_index >= len(available_files):
        return None, "[]", f"No file available for index {file_index} in {selected_subdir}", env_data, groups_data
    
    file_path = available_files[file_index]
    try:
        img, npy = load_image_and_npy(file_path, 'bev')
        env_data.img1 = img
        env_data.npy1 = npy

        # Clear all groups' points when new base image is loaded
        for group in groups_data:
            group.points = []
        
        filename = os.path.basename(file_path)
        status = f"Loaded: {selected_subdir}/{filename}"
        return img, status, env_data, groups_data
    except Exception as e:
        return None, f"Error loading {file_path}: {e}", env_data, groups_data


# ── Inference (single trajectory, one-step) ───────────────────────────────────
def run_inference(env_data, groups_data):
    """Run model inference"""
    if model is None:
        return None, "Please load a model first"
    if env_data.npy1 is None:
        return None, "Please upload the BEV map first"

    group = groups_data[0]
    if group.npy2 is None:
        return None, "No tangent map — process or upload a trajectory first"
    if group.npy3 is None:
        return None, "No distance map — process or upload a trajectory first"

    try:
        input_data = {
            "bev_map":       torch.from_numpy(env_data.npy1.copy()).to(torch_dtype),
            "tangent_init":  torch.from_numpy(group.npy2.copy()).to(torch_dtype),
            "distance_init": torch.from_numpy(group.npy3.copy()).to(torch_dtype),
        }

        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = model([input_data])
            print(f"compute_time: {time.perf_counter() - t0:.3f}s")

        assert len(outputs) == 1
        output = outputs[0]
        output_regression = output["regression"][0].to('cpu')
        pred = peroidic_np(np.array(output_regression.detach(), dtype=np.float32), norm=False, abs=False)
        tangent_init = input_data["tangent_init"].numpy()
        tangent_init_polar = cartesian_to_polar(tangent_init[..., 0], tangent_init[..., 1])[1] / np.pi
        tangent_map_pred_polar = peroidic_np(get_pred(tangent_init_polar, pred), norm=False, abs=False)
        tangent_map_pred_x, tangent_map_pred_y = polar_to_cartesian(np.ones_like(tangent_map_pred_polar), tangent_map_pred_polar * np.pi)
        tangent_map_pred = np.stack([tangent_map_pred_x, tangent_map_pred_y], axis=-1)
        result_np = tangent_map_pred
        result_image = Image.fromarray(
            (np.clip(visulizater.tangent_vis(result_np, pltcm='hsv'), None, 1) * 255
             ).astype(np.uint8)
        )
        return result_image, "Inference completed successfully"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during inference: {e}"


# ── Gradio interface ──────────────────────────────────────────────────────────
with gr.Blocks(title="Orientation Field", theme=gr.themes.Soft()) as demo:

    env_data_state    = gr.State(Environment())
    groups_data_state = gr.State([TrajectoryGroup(0)])   # single trajectory

    gr.Markdown(
        """
        # 🧭 Orientation Field
        Interactive demo for
        [OriField: Learning Orientation Field for OSM-Guided Autonomous Navigation](https://github.com/IMRL/OriField)
        """
    )

    # ── Model status bar ──────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=4):
            model_status = gr.Textbox(
                label="Model Status", 
                value=initial_status,
                interactive=False,
                container=True
            )
        with gr.Column(scale=1):
            refresh_status_btn = gr.Button("🔄 Check Status", size="sm")

    # ── Instructions ──────────────────────────────────────────────────────────
    with gr.Accordion("📋 Instructions", open=False):
        gr.Markdown(
            """
            1. **Model Loading** — loads automatically from `--config-file` or default paths.
            2. **Upload Environment** — upload a BEV `.npy` file or pick one from the example sliders.
            3. **Configure Trajectory** — upload pre-computed tangent/distance maps  
               *or* click ≥ 3 points on the BEV image and press **Process Trajectory**.
            4. **Run Inference** — click **🚀 Run Inference** for a direct one-step prediction.
            5. **Check Status** — if the inference button stays disabled, click **Check Status**.
            """
        )

    # ── Top row ───────────────────────────────────────────────────────────────
    with gr.Row():
        # BEV map display
        with gr.Column(scale=2):
            gr.Markdown("### 🌍 Environment (BEV Map)")
            img1_display = gr.Image(label="BEV Map", type="pil", height=400)

        # File picker
        with gr.Column(scale=2):
            gr.Markdown("### 📁 File Selection")
            if available_subdirs:
                gr.Markdown("**Example Files**")
                subdir_dropdown = gr.Dropdown(
                    choices=available_subdirs,
                    value=available_subdirs[0] if available_subdirs else None,
                    label="Subdirectory",
                    interactive=True
                )
                file_slider = gr.Slider(
                    minimum=0,
                    maximum=max(0, len(current_available_files) - 1),
                    value=0,
                    step=1,
                    label=f"File Index (0-{max(0, len(current_available_files) - 1)})",
                    interactive=True
                )
                example_info = gr.Textbox(
                    label="Current Selection", 
                    value=f"{available_subdirs[0] if available_subdirs else 'None'}/{os.path.basename(current_available_files[0]) if current_available_files else 'None'}",
                    interactive=False,
                    lines=1
                )
            else:
                gr.Markdown("*No example subdirectories found*")
                subdir_dropdown = None
                file_slider = None
                example_info = None

            gr.Markdown("**Manual Upload**")
            img1_upload = gr.File(
                label="Upload Custom .npy or .bin File", 
                file_types=[".npy", ".bin"],
                height=80
            )

        # Model controls
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ Model Controls")
            gr.Markdown("*Single-step network prediction*")
            submit_btn = gr.Button(
                "🚀 Run Inference", 
                variant="primary", 
                size="lg",
                interactive=True
            )
            result_status = gr.Textbox(
                label="Status", 
                interactive=False,
                lines=3,
                max_lines=5,
                placeholder="Ready for input..."
            )

    # ── Results ───────────────────────────────────────────────────────────────
    with gr.Accordion("📊 Result", open=False, visible=False) as results_accordion:
        result_image = gr.Image(
            label="Generated Orientation Field", type="pil", height=400
        )

    # ── Trajectory section header ─────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 🛤️ Trajectory Configuration")
        with gr.Column(scale=1):
            clear_data_btn = gr.Button(
                "❌ Clear Trajectory Data", variant="secondary", size="lg"
            )

    interactive_updaters = []

    with gr.Row():
        with gr.Column():
            img2_disp_1, img3_disp_1, interactive_1, _ = create_trajectory_group(
                0, env_data_state, groups_data_state
            )
            interactive_updaters.append(interactive_1)

    # ── Model status events ───────────────────────────────────────────────────
    def check_model_status():
        """Check current model loading status and return UI updates"""
        global model, model_loading, model_status_message
        is_loaded = model is not None and not model_loading
        print(f"Status check - Model loaded: {model is not None}, Loading: {model_loading}, Status: {model_status_message}")
        return [
            model_status_message,  # model_status
            gr.Button(interactive=is_loaded)  # submit_btn
        ]

    # Manual status refresh button
    refresh_status_btn.click(
        check_model_status,
        outputs=[model_status, submit_btn]
    )
    
    # ── BEV map upload ────────────────────────────────────────────────────────
    def upload_img1(file, env_data, groups_data):
        if file:
            if file.name.endswith('.npy'):
                img, npy = load_image_and_npy(file.name, 'bev')
                env_data.img1 = img
                env_data.npy1 = npy
            elif file.name.endswith('.bin'):
                # Handle .bin files if needed (e.g., convert to BEV map)
                points = np.fromfile(file.name, dtype=np.float32).reshape(-1, 4)
                labels = np.zeros(points.shape[0], dtype=np.int32)  # Placeholder labels
                points[:, 2] += 1.73
                pointcloud_new = points.copy()
                pointcloud_new[:, 0] = -points[:, 1]
                pointcloud_new[:, 1] = points[:, 0]
                points = pointcloud_new
                bev_map = bev.rgb_label_map(points[:, :4], labels[:, None], np.array([-32.0, 32.0, -32.0, 32.0, -2.5, 3.5]), 0.16)
                bev_map = bev_map.reshape((400, 400, -1)).copy()
                npy = bev_map[..., :-1]
                img = Image.fromarray((np.clip(visulizater.naive_vis(npy), 0, 1) * 255).astype(np.uint8))
                env_data.img1 = img
                env_data.npy1 = npy
            else:
                return None, "Unsupported file type. Please upload a .npy or .bin file.", env_data, groups_data
            for group in groups_data:
                group.points = []
            return [img] + [img] * len(interactive_updaters) + [env_data, groups_data]
        return [None] * (1 + len(interactive_updaters)) + [env_data, groups_data]

    img1_upload.change(upload_img1,
        inputs=[img1_upload, env_data_state, groups_data_state],
        outputs=[img1_display] + interactive_updaters + [env_data_state, groups_data_state])

    # ── Example file slider events ────────────────────────────────────────────
    if available_subdirs:
        def update_file_slider_range(selected_subdir):
            """Update file slider range when subdirectory changes"""
            global current_available_files
            current_available_files = find_npy_files_in_subdir(example_dir, selected_subdir)
            max_files = max(0, len(current_available_files) - 1)
            
            return [
                gr.Slider(
                    minimum=0,
                    maximum=max_files,
                    value=0,
                    step=1,
                    label=f"Select File (0-{max_files})",
                    info=f"Found {len(current_available_files)} .npy files",
                    interactive=True
                ),
                f"Subdirectory: {selected_subdir}, File: {os.path.basename(current_available_files[0]) if current_available_files else 'None'}"
            ]
        
        def load_from_sliders(selected_subdir, file_index, env_data, groups_data):
            """Load the selected example file during sliding"""
            subdir_index = available_subdirs.index(selected_subdir) if selected_subdir in available_subdirs else 0
            img, status, env_data, groups_data = load_npy_from_selection(subdir_index, file_index, available_subdirs, env_data, groups_data)
            
            # Update interactive images if img1 was loaded
            if img is not None:
                return [img] + [img] * len(interactive_updaters) + [status, env_data, groups_data]
            return [None] * (1 + len(interactive_updaters)) + [status, env_data, groups_data]
        
        # Connect subdirectory dropdown events
        subdir_dropdown.change(
            update_file_slider_range,
            inputs=[subdir_dropdown],
            outputs=[file_slider, example_info]
        )
        # Connect file slider events to load immediately during sliding
        file_slider.change(
            load_from_sliders,
            inputs=[subdir_dropdown, file_slider, env_data_state, groups_data_state],
            outputs=[img1_display] + interactive_updaters + [example_info, env_data_state, groups_data_state]
        )
    
    # ── Inference button ──────────────────────────────────────────────────────
    def run_inference_with_results(env_data, groups_data):
        global model, model_loading

        # Always check current status first
        current_status = check_model_status()
        print(f"Inference button clicked - Model status: {current_status}")
        
        if model_loading:
            return (None,
                    "🔄 Model is still loading. Please wait and try again...",
                    gr.Accordion(open=False, visible=False),
                    env_data, groups_data)
        if model is None:
            return (None,
                    "❌ No model loaded. Please specify --config-file.",
                    gr.Accordion(open=False, visible=False),
                    env_data, groups_data)
        try:
            result_img, status = run_inference(env_data, groups_data)
            if result_img is not None:
                return (result_img, status,
                        gr.Accordion(open=True, visible=True),
                        env_data, groups_data)
            return (None,
                    status or "❌ Inference completed but produced no result.",
                    gr.Accordion(open=False, visible=False),
                    env_data, groups_data)
        except Exception as e:
            return (None, f"❌ Inference failed: {e}",
                    gr.Accordion(open=False, visible=False),
                    env_data, groups_data)

    submit_btn.click(run_inference_with_results,
        inputs=[env_data_state, groups_data_state],
        outputs=[result_image, result_status, results_accordion,
                 env_data_state, groups_data_state])

    # ── Clear trajectory data ─────────────────────────────────────────────────
    def clear_trajectory_data(env_data, groups_data):
        g = groups_data[0]
        g.img2 = g.npy2 = g.img3 = g.npy3 = None
        return "✅ Trajectory data cleared", None, None, env_data, groups_data

    clear_data_btn.click(clear_trajectory_data,
        inputs=[env_data_state, groups_data_state],
        outputs=[result_status, img2_disp_1, img3_disp_1,
                 env_data_state, groups_data_state])


if __name__ == "__main__":
    print("Starting Gradio interface...")
    print("Tip: after the model loads, click 'Check Status' to enable the inference button.")
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        debug=False,
        show_error=True,
        quiet=False,
    )