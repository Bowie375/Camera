import os

import numpy as np
import cv2
from matplotlib import pyplot as plt

from data.read_meta import read_meta
from utils import PointCloudProcessor
from stereo_matcher import match_multichannel_rowwise_fast
from common import (
    disp_to_depth, 
    rectify_and_crop, 
    depth_to_pointcloud, 
    build_multichannel_descriptor,
)


def task1(meta_file_path: str, single_shot: bool = True, single_shot_number: int = 0, visualize: bool = True):
    """
    Args:
        meta_file_path (str): Path to meta file.
        single_shot (bool): Whether to run single-shot mode.
        single_shot_number (int): Which single-shot to run.
        visualize (bool): Whether to visualize the plotted results.
    """

    meta = read_meta(meta_file_path)

    if single_shot:
        left_img_path = os.path.join(
            os.path.dirname(meta_file_path), 
            "single_shot", 
            f'cam_L_{single_shot_number:04d}.png'
        )
        rgb_img_path = os.path.join(
            os.path.dirname(meta_file_path), 
            "rgb",
            "cam_L_0008.png"
        )
        proj_img_path = f"data/patterns/{single_shot_number:04d}.png"
        left_img_list = [np.array(cv2.imread(left_img_path)), np.array(cv2.imread(rgb_img_path))]
        proj_img_list = [np.array(cv2.imread(proj_img_path))]
    else:
        left_img_list = []
        proj_img_list = []
        for i in range(32):
            left_img_path = os.path.join(
                os.path.dirname(meta_file_path), 
                "alacarte_32_camL",
                f"cam_L_{i:04d}.png"
            )
            proj_img_path = f"data/patterns/alacarte_32/{i:04d}.png"
            left_img_list.append(np.array(cv2.imread(left_img_path)))
            proj_img_list.append(np.array(cv2.imread(proj_img_path)))
        rgb_img_path = os.path.join(
            os.path.dirname(meta_file_path), 
            "rgb",
            "cam_L_0008.png"
        )
        left_img_list.append(np.array(cv2.imread(rgb_img_path)))

    left_rect, proj_rect, mask, P1_crop, _, R1, _, _, _ = rectify_and_crop(meta, left_img_list, proj_img_list)
    
    p, disp = None, None
    if single_shot:
        p = os.path.join(os.path.dirname(meta_file_path), f"disp_{single_shot_number:04d}.npy")
    else:
        p = os.path.join(os.path.dirname(meta_file_path), "disp_alacarte32.npy")

    if os.path.exists(p):
        disp = np.load(p)
    else:
        left_desc = build_multichannel_descriptor(left_rect[:-1])
        proj_desc = build_multichannel_descriptor(proj_rect)

        print("<------- Compute disparity ------->")
        import time
        t0 = time.time()
        # disp = match_multichannel_rowwise(left_desc, proj_desc, mask, max_disp=800, win=100)
        disp = match_multichannel_rowwise_fast(left_desc, proj_desc, mask, max_disp=800, win=100)
        t1 = time.time()
        print(f"Matching time: {t1-t0:.3f}s\n")
        np.save(p, disp)

    depth = disp_to_depth(disp, P1_crop, np.linalg.norm(meta['T']))

    #------------------------------------------------------------------------------------------------------#

    ## plot a curve showing (x, disp[y, x]), (x, depth[y, x])
    horizon = 500
    plt.figure(figsize=(12, 4))
    plt.subplot(121)
    plt.plot(np.arange(disp.shape[1]), disp[horizon, :], 'b')
    plt.title("Disparity")
    plt.subplot(122)
    plt.plot(np.arange(depth.shape[1]), depth[horizon, :], 'r')
    plt.title("Depth")
    plt.show(block=visualize)

    ## plot the matching result
    h1, w1 = left_rect[0].shape[:2]
    h2, w2 = proj_rect[0].shape[:2]
    canvas_h = h1 + h2
    canvas_w = max(w1, w2)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = np.repeat(np.expand_dims(left_rect[0], -1), 3, axis=2)
    # canvas[500:800, 500:1200] = np.array([255, 0, 0])
    canvas[h1:h1+h2, :w2] = np.repeat(np.expand_dims(proj_rect[0], -1), 3, axis=2)
    plt.figure(figsize=(8, 12))
    plt.title(f"Row Matching Result, horizon={horizon}")
    plt.imshow(canvas)
    plt.axis('off')

    cmap = plt.cm.viridis
    for i in range(0, w1, 50):
        x1, y1 = i, horizon
        if np.isnan(disp[y1, x1]):
            continue
        x2, y2 = x1 - disp[y1, x1], y1 + h1
        plt.plot([x1, x2], [y1, y2], color=cmap(x1/w1), linewidth=2)   # red line

    plt.show(block=visualize)

    ## plot the disparity map and depth map
    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(disp, cmap="turbo")
    plt.colorbar()
    plt.title("Disparity Map")
    plt.subplot(122)
    plt.imshow(depth, cmap="gray")
    plt.colorbar()
    plt.title("Depth Map")
    plt.show(block=visualize)

    #------------------------------------------------------------------------------------------------------#

    ## Export point cloud
    pcd, xs, ys = depth_to_pointcloud(depth, P1_crop)

    pixel_orig = pcd @ R1 @ meta["KL"].T
    pixel_horm = (pixel_orig[:, :2] / pixel_orig[:, 2:3]).astype(np.int32)
    pixel_mask = (pixel_horm[:, 0] >= 0) & (pixel_horm[:, 0] < w1) & (pixel_horm[:, 1] >= 0) & (pixel_horm[:, 1] < h1)

    depth_orig = np.zeros(left_img_list[-1].shape[:2], dtype=np.float32)
    depth_orig[pixel_horm[:, 1][pixel_mask], pixel_horm[:, 0][pixel_mask]] = pixel_orig[:, 2][pixel_mask]

    plt.figure(figsize=(12, 8))
    plt.subplot(121)
    plt.imshow(left_img_list[-1])
    plt.title("Original Image")
    plt.subplot(122)
    plt.imshow(depth_orig, cmap="gray")
    plt.title("Depth Map")
    plt.show(block=visualize) 

    pcd_orig, xs, ys = depth_to_pointcloud(depth_orig, meta["KL"])
    colors = cv2.cvtColor(left_img_list[-1], cv2.COLOR_BGR2RGB)[ys, xs] / 255.0

    pcp = PointCloudProcessor(pcd_orig, colors, num_points=4096)
    print("<------- Process point cloud ------->")
    pcd_processed, idx = pcp.process()

    os.makedirs("tmp/pcd", exist_ok=True)
    if single_shot:
        depth_path = os.path.join(os.path.dirname(meta_file_path), f"depth_{single_shot_number:04d}.npy")
        pcd_path = os.path.join(os.path.dirname(meta_file_path), f"pcd_{single_shot_number:04d}.ply")
    else:
        depth_path = os.path.join(os.path.dirname(meta_file_path), "depth_alacarte32.npy")
        pcd_path = os.path.join(os.path.dirname(meta_file_path), "pcd_alacarte32.ply")

    np.save(depth_path, depth_orig)
    pcp.save_pcd(pcd_processed, pcd_path)


if __name__ == '__main__':
    meta_file_path = 'data/objects/book/meta.npy'
    task1(meta_file_path, single_shot=False, single_shot_number=1, visualize=False)