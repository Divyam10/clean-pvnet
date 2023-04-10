from ast import While
from lib.config import cfg, args
import numpy as np
import os
import cv2
import pyrealsense2
from realsense_depth import *
from pynput import keyboard

def run_demo(color_frame):
    from lib.datasets import make_data_loader
    from lib.visualizers import make_visualizer
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils.net_utils import load_network
    import glob
    from PIL import Image

    torch.manual_seed(0)
    meta = np.load(os.path.join(cfg.demo_path, 'meta.npy'), allow_pickle=True).item()
    # demo_images = glob.glob(cfg.demo_path + '/*jpg')
    demo_images = [color_frame]
    network = make_network(cfg).cuda()
    print(cfg.model_dir)
    load_network(network, cfg.model_dir, epoch=cfg.test.epoch)
    network.eval()

    visualizer = make_visualizer(cfg)

    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    for demo_image in demo_images:
        # demo_image = np.array(Image.open(demo_image)).astype(np.float32)
        inp = (((demo_image/255.)-mean)/std).transpose(2, 0, 1).astype(np.float32)
        inp = torch.Tensor(inp[None]).cuda()
        with torch.no_grad():
            output = network(inp)
        visualizer.visualize_demo(output, inp, meta)

def on_press(key):
    if key == keyboard.Key.esc:
        return False  # stop listener
    try:
        k = key.char  # single-char keys
    except:
        k = key.name  # other keys
    if k in ['r']:  # keys of interest
        # self.keys.append(k)  # store it in global-like variable
        print('Key pressed: ' + k)
        print("Process Restarting")
        dc = DepthCamera()
        ret, depth_frame, color_frame = dc.get_frame()
        dc.release()
        globals()['run_'+args.type](color_frame)
        

if __name__ == '__main__':


    listener = keyboard.Listener(on_press=on_press)
    listener.start()  # start to listen on a separate thread
    listener.join() 
    

    

    