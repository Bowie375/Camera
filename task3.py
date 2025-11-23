import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from data.read_meta import read_meta
from common import depth_to_pointcloud
from utils import PointCloudProcessor

def warp_to_left(
    KL: np.ndarray,
    KR: np.ndarray,
    cam_R_pose: np.ndarray,
    depth_L: np.ndarray,
    img_R: cv2.Mat,
):
    """
    Args:
        KL (np.ndarray): Intrinsics of the left camera.
        KR (np.ndarray): Intrinsics of the right camera.
        cam_R_pose (np.ndarray): Pose of the right camera(relatively to the left camera).
        depth_L (np.ndarray): Depth map of the left camera.
        img_R (cv2.Mat): RGB image of the right camera.

    Returns:
        np.ndarray: RGB image of the left camera.
    """

    h_L, w_L = depth_L.shape
    img_L = np.zeros((h_L, w_L, 3), dtype=np.uint8)

    pcd_L, xs_L, ys_L = depth_to_pointcloud(depth_L, KL)
    
    pcd_L_horm = np.hstack([pcd_L, np.ones((pcd_L.shape[0], 1))])
    cam_L_world = np.array([
        [-1., 0.0, 0.0, 0.0],
        [0.0, -1., 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    cam_L_cam_R = np.linalg.inv(cam_R_pose)
    pcd_R_horm = (cam_L_world.T @ cam_L_cam_R @ cam_L_world @ pcd_L_horm.T).T
    pcd_R = pcd_R_horm[:, :3] / pcd_R_horm[:, 3:4]

    pixel_R = np.dot(KR, pcd_R.T).T.astype(np.float32)
    zero_depth_mask = (pixel_R[:, 2] > 1e-6)
    pixel_R[zero_depth_mask, :2] /= pixel_R[zero_depth_mask, 2:3]
    xs_R, ys_R = pixel_R[:, 0][zero_depth_mask], pixel_R[:, 1][zero_depth_mask]

    h_R, w_R = img_R.shape[:2]
    valid_mask = (xs_R >= 0) & (xs_R < w_R) & (ys_R >= 0) & (ys_R < h_R)
    xs_R = xs_R[valid_mask].astype(np.int32)
    ys_R = ys_R[valid_mask].astype(np.int32)
    xs_L = xs_L[zero_depth_mask][valid_mask].astype(np.int32)
    ys_L = ys_L[zero_depth_mask][valid_mask].astype(np.int32)
    img_L[ys_L, xs_L] = img_R[ys_R, xs_R]

    return img_L

def task3(meta_file_path: str, single_shot: bool = True, single_shot_number: int = 0, visualize: bool = True):
    """
    Args:
        meta_file_path (str): Path to meta file.
        single_shot (bool): Whether to run single-shot mode.
        single_shot_number (int): Which single-shot to run.
        visualize (bool): Whether to visualize the plotted results.
    """

    rgb_R_orig_path = os.path.join(os.path.dirname(meta_file_path), 'rgb', 'cam_R_0008.png')
    rgb_L_orig_path = os.path.join(os.path.dirname(meta_file_path), 'rgb', 'cam_L_0008.png')
    depth_L_path = os.path.join(os.path.dirname(meta_file_path), 'depth_alacarte32.npy')
    pcd_L_path = os.path.join(os.path.dirname(meta_file_path), 'pcd_alacarte32.ply')
    if single_shot:
        depth_L_path = os.path.join(os.path.dirname(meta_file_path), f'depth_{single_shot_number:04d}.npy')
        pcd_L_path = os.path.join(os.path.dirname(meta_file_path), f'pcd_{single_shot_number:04d}.ply')

    meta = read_meta(meta_file_path)
    depth_L = np.load(depth_L_path)
    rgb_R_orig = cv2.cvtColor(cv2.imread(rgb_R_orig_path), cv2.COLOR_BGR2RGB)
    rgb_L_orig = cv2.cvtColor(cv2.imread(rgb_L_orig_path), cv2.COLOR_BGR2RGB)
    cam_R_pose = np.load(os.path.join(os.path.dirname(meta_file_path), 'cam_R_pose.npy'))

    print("<------- Render Right Image ------->")
    print(f"cam_R_pose: \n{cam_R_pose}\ncam_intrinsics: \n{meta['KR']}\n")

    pcp = PointCloudProcessor(pcd_L_path)    
    rgb_R, depth_R = pcp.render(
        camera_extrinsics=cam_R_pose.copy(),
        camera_intrinsics=meta['KR'],
        out_width=rgb_R_orig.shape[1],
        out_height=rgb_R_orig.shape[0],
    )

    print("<------- Warp Left Image ------->\n") 
    print(f"cam_R_pose: \n{cam_R_pose}\n")
    
    rgb_L_wrap = warp_to_left(meta['KL'], meta['KR'], cam_R_pose, depth_L, rgb_R_orig)

    #------------------------------------------------------------------------------------------------------#

    # Plot the rendered RGB and depth images
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_R_orig)
    plt.title("Original cam_R RGB")

    plt.subplot(1, 3, 2)
    plt.imshow(rgb_R)
    plt.title("Rendered cam_R RGB")

    plt.subplot(1, 3, 3)
    plt.imshow(depth_R)
    plt.title("Rendered cam_R Depth")

    plt.show(block=visualize)

    # Plot the warped RGB image
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(rgb_R_orig)
    plt.title("Right Image")

    plt.subplot(1, 3, 2)
    plt.imshow(rgb_L_wrap)
    plt.title("Warped Left Image")

    plt.subplot(1, 3, 3)
    plt.imshow(rgb_L_orig)
    plt.title("Original Left Image")

    plt.show(block=visualize)

    #------------------------------------------------------------------------------------------------------#

    # save the rendered RGB image
    rgb_export_path = os.path.join(os.path.dirname(meta_file_path), 'cam_R_rgb_alacarte32.png')
    if single_shot:
        rgb_export_path = os.path.join(os.path.dirname(meta_file_path), f'cam_R_rgb_{single_shot_number:04d}.png')
    cv2.imwrite(rgb_export_path, cv2.cvtColor(rgb_R, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    meta_file_path = 'data/objects/book/meta.npy'
    task3(meta_file_path, single_shot=False, single_shot_number=0, visualize=True)