from ast import While
from lib.config import cfg, args
import numpy as np
import os
import cv2
# import pyrealsense2
from realsense_depth import *
# from pynput import keyboard


from lib.datasets import make_data_loader
from lib.visualizers import make_visualizer
# import tqdm
import torch
from lib.networks import make_network
from lib.utils.net_utils import load_network
# import glob
from PIL import Image
import matplotlib.pyplot as plt

from lib.utils import img_utils
import matplotlib.patches as patches
import trimesh
from lib.utils.pvnet import pvnet_pose_utils
import matplotlib.cm as cm
import time

def render(img, pose, K, obj):
    # get the 3d points of the object
    points = obj.vertices
    # get the 3d points in camera coordinates
    points = np.dot(pose[:3, :3], points.T).T + pose[:3, 3]
    # get the 2d points in image coordinates
    points = np.dot(K, points.T).T
    points = points[:, :2] / points[:, 2:]
    # get the 2d points in image coordinates
    points = np.round(points).astype(np.int32)
    # print(points.shape)
    # get the color of the object
    # print(points.shape)
    colors = cm.get_cmap()(obj.visual.vertex_colors[:, 0])[:, :3]

    points_x = np.clip(points[:, 0], 0, img.shape[1] - 1)
    points_y = np.clip(points[:, 1], 0, img.shape[0] - 1)
    img[ points_y, points_x] = [0.477504*255, 0.821444*255, 0.318195* 255] 
    # print(points[1].shape, points[0].shape)
    # img[points[1], points[0]] = [0.477504, 0.821444, 0.318195] * 255
    # for i, point in enumerate(points):
    #     if 0 <= point[0] < img.shape[1] and 0 <= point[1] < img.shape[0]:
    #         img[point[1], point[0]] = colors[i] * 255
          
    return img

# convert the 3d points to 2d points

obj = trimesh.load('/home/ai/pose_est/clean-pvnet-1.10/data/welding/model.ply')
torch.manual_seed(0)
meta = np.load(os.path.join(cfg.demo_path, 'meta.npy'), allow_pickle=True).item()
# demo_images = glob.glob(cfg.demo_path + '/*jpg')
# demo_images = [color_frame]
network = make_network(cfg).cuda()
print(cfg.model_dir),
load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
network.eval()

dc = DepthCamera()

# continue loop till q is pressed

while True:
    time_start = time.time()

    ret, depth_frame, color_frame = dc.get_frame()

    color_frame_bgr = color_frame
    # visualizer = make_visualizer(cfg)
    color_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('color_frame.png', color_frame)
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    # for demo_image in demo_images:
        # demo_image = np.array(Image.open(demo_image)).astype(np.float32)
    inp = (((color_frame/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
    inp = torch.Tensor(inp[None]).cuda()
    with torch.no_grad():
        output = network(inp)
    inp = img_utils.unnormalize_img(inp[0], mean, std).permute(1, 2, 0)
    kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()

    kpt_3d = np.array(meta['kpt_3d'])
    K = np.array(meta['K'])

    pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

    corner_3d = np.array(meta['corner_3d'])
    corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
    # print(pose_pred)
    # _, ax = plt.subplots(1)
    # ax.imshow(inp)
    # ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='b'))
    # ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='b'))
    ## Add the above patches in cv2 image

    cv2.polylines(color_frame_bgr, [corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]].astype(np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(color_frame_bgr, [corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]].astype(np.int32)], True, (255, 0, 0), 2)

    color_frame_bgr = render(color_frame_bgr, pose_pred, K, obj)
    cv2.imshow('color_frame', color_frame_bgr)
    # plt.show(block=False)
    # plt.pause(0.3)
    # plt.close()
    # cv2.imshow('color_frame', inp)
    print('FPS: ', 1/(time.time() - time_start))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()

        break
