import os
import sys
import json
import h5py
import argparse
import importlib
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
from shutil import copyfile
from plyfile import PlyData, PlyElement
import matplotlib
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from utils.pc_utils import write_ply_rgb, write_oriented_bbox
from utils.box_util import get_3d_box, box3d_iou
from models.refnet import RefNet
from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.dataset import ScannetReferenceDataset
from lib.solver import Solver
from lib.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.loss_helper import get_loss
from lib.eval_helper import get_eval
from lib.config import CONF

# data
SCANNET_ROOT = "/home/adl4cv/yoonha/ScanRefer/data/scannet/scans" # TODO point this to your scannet data
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply") # scene_id, scene_id 
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt") # scene_id, scene_id 
SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, config, augment):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_color=args.use_color, 
        use_height=(not args.no_height),
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    return dataset, dataloader

def get_model(args):
    # load model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = RefNet(
        num_class=DC.num_class,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        num_proposal=args.num_proposals,
        input_feature_dim=input_channels,
        use_brnet=args.use_brnet,
        use_self_attn=args.use_self_attn,
        use_cross_attn=args.use_cross_attn,
        use_dgcnn=args.use_dgcnn
    ).cuda()

    path = os.path.join(CONF.PATH.OUTPUT, args.folder, "model.pth")
    model.load_state_dict(torch.load(path), strict=False)
    model.eval()

    return model

def get_scanrefer(args):
    scanrefer = SCANREFER_TRAIN if args.use_train else SCANREFER_VAL
    all_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))
    if args.scene_id:
        assert args.scene_id in all_scene_list, "The scene_id is not found"
        scene_list = [args.scene_id]
    else:
        scene_list = sorted(list(set([data["scene_id"] for data in scanrefer])))

    scanrefer = [data for data in scanrefer if data["scene_id"] in scene_list]

    return scanrefer, scene_list

def visualize_attn(args):
    dump_dir = os.path.join(CONF.PATH.OUTPUT, args.folder, "attn", args.scene_id)
    os.makedirs(dump_dir, exist_ok=True)
    # init training dataset
    print("preparing data...")
    scanrefer, scene_list = get_scanrefer(args)

    # dataloader
    _, dataloader = get_dataloader(args, scanrefer, scene_list, "val", DC, False)

    # model
    model = get_model(args)

    print("visualizing attention weights...")
    for data in tqdm(dataloader):
        for key in data:
            data[key] = data[key].cuda()
        with torch.no_grad():
            data = model.lang(data)
            # from inputs
            ids = data['scan_idx'].detach().cpu().numpy()
            point_clouds = data['point_clouds'].cpu().numpy()
            batch_size = point_clouds.shape[0]
            colored_string = ''
            for i in range(batch_size):
                # basic info
                idx = ids[i]
                token = scanrefer[idx]["token"]
                cmap = matplotlib.cm.Blues
                template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
                for word, color in zip(token, data["attn_weight"][i].tolist()):
                    color = matplotlib.colors.rgb2hex(cmap(color)[:3])
                    colored_string += template.format(color, '&nbsp' + word + '&nbsp')
                colored_string += """</br>"""
        # save in an html file and open in browser
        with open(os.path.join(dump_dir, 'attention.html'), 'a') as f:
            f.write(colored_string)
    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, help="Folder containing the model", required=True)
    parser.add_argument("--gpu", type=str, help="gpu", default="0")
    parser.add_argument("--scene_id", type=str, help="scene id", default="")
    parser.add_argument("--batch_size", type=int, help="batch size", default=1)
    parser.add_argument('--num_points', type=int, default=40000, help='Point Number [default: 40000]')
    parser.add_argument('--num_proposals', type=int, default=256, help='Proposal number [default: 256]')
    parser.add_argument('--num_scenes', type=int, default=-1, help='Number of scenes [default: -1]')
    parser.add_argument('--no_height', action='store_true', help='Do NOT use height signal in input.')
    parser.add_argument('--no_nms', action='store_true', help='do NOT use non-maximum suppression for post-processing.')
    parser.add_argument('--use_train', action='store_true', help='Use the training set.')
    parser.add_argument('--use_color', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_normal', action='store_true', help='Use RGB color in input.')
    parser.add_argument('--use_multiview', action='store_true', help='Use multiview images.')
    parser.add_argument("--use_brnet", action="store_true", help="Use BRNet for object detection.")
    parser.add_argument("--use_self_attn", action="store_true", help="Use self attention for lang features.")
    parser.add_argument("--use_cross_attn", action="store_true", help="Use cross attention with visual and lang features.")
    parser.add_argument("--use_dgcnn", action="store_true", help="Use DGCNN for visual features.")
    parser.add_argument("--fuse_before", action="store_true", help="Fuse before DGCNN.")
    args = parser.parse_args()

    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    visualize_attn(args)
