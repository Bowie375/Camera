import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data.read_meta import read_meta
from utils import PointCloudProcessor

def task3(meta_file_path: str, single_shot: bool = True, single_shot_number: int = 0, visualize: bool = True):
    """
    Args:
        meta_file_path (str): Path to meta file.
        single_shot (bool): Whether to run single-shot mode.
        single_shot_number (int): Which single-shot to run.
        visualize (bool): Whether to visualize the plotted results.
    """

    rgb_orig_path = os.path.join(os.path.dirname(meta_file_path), 'rgb', 'cam_R_0008.png')
    pcd_path = os.path.join(os.path.dirname(meta_file_path), 'pcd_alacarte32.ply')
    if single_shot:
        pcd_path = os.path.join(os.path.dirname(meta_file_path), f'pcd_{single_shot_number:04d}.ply')

    meta = read_meta(meta_file_path)
    rgb_orig = cv2.cvtColor(cv2.imread(rgb_orig_path), cv2.COLOR_BGR2RGB)
    cam_R_pose = np.load(os.path.join(os.path.dirname(meta_file_path), 'cam_R_pose.npy'))

    pcp = PointCloudProcessor(pcd_path)
    rgb, depth = pcp.render(
        camera_extrinsics=cam_R_pose,
        camera_intrinsics=meta['KR'],
        out_width=rgb_orig.shape[1],
        out_height=rgb_orig.shape[0],
    )

    # Plot the RGB and depth images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_orig)
    plt.title("Original RGB")

    plt.subplot(1, 3, 2)
    plt.imshow(rgb)
    plt.title("Rendered RGB")

    plt.subplot(1, 3, 3)
    plt.imshow(depth)
    plt.title("Depth")

    plt.show(block=visualize)

    # save the rendered RGB and depth images
    rgb_export_path = os.path.join(os.path.dirname(meta_file_path), 'cam_R_rgb_alacarte32.png')
    depth_export_path = os.path.join(os.path.dirname(meta_file_path), 'cam_R_depth_alacarte32.png')
    if single_shot:
        rgb_export_path = os.path.join(os.path.dirname(meta_file_path), f'cam_R_rgb_{single_shot_number:04d}.png')
        depth_export_path = os.path.join(os.path.dirname(meta_file_path), f'cam_R_depth_{single_shot_number:04d}.png')
    cv2.imwrite(rgb_export_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(depth_export_path, (depth / depth.max() * 255.0).astype(np.uint8))

if __name__ == '__main__':
    meta_file_path = 'data/objects/book/meta.npy'
    task3(meta_file_path, single_shot=False, single_shot_number=0, visualize=True)