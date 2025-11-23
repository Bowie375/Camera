import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from common import depth_to_pointcloud
from data.read_meta import read_meta

def match_features(img_L: cv2.Mat, img_R: cv2.Mat):
    orb = cv2.ORB_create(5000)
    k1, d1 = orb.detectAndCompute(img_L, None)
    k2, d2 = orb.detectAndCompute(img_R, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(d1, d2)
    matches = sorted(matches, key=lambda x: x.distance)

    pts_L = np.float32([k1[m.queryIdx].pt for m in matches])
    pts_R = np.float32([k2[m.trainIdx].pt for m in matches])
    return pts_L, pts_R, matches

def estimate_right_camera_pose(
    KL: np.ndarray, 
    KR: np.ndarray, 
    DR: np.ndarray, 
    depth_L: np.ndarray, 
    img_L: cv2.Mat, 
    img_R: cv2.Mat, 
    crop_offset_L: np.ndarray,
    crop_offset_R: np.ndarray,
    visualize=True
):
    """
    Estimate the right camera pose given the left camera pose, depth map in left camera, and image pair.
    
    Args:
        KL: intrinsics of the left camera
        KR: intrinsics of the right camera
        DR: distortion coefficients of the right camera
        depth_L: depth map in the left camera
        img_L: left image
        img_R: right image
        crop_offset_L: offset of the crop in the left image
        crop_offset_R: offset of the crop in the right image
        visualize: whether to visualize the matching result
        
    Returns:
        R: rotation matrix of the right camera (relative to the left camera)
        t: translation vector of the right camera (relative to the left camera)
    """

    # 1. Match features
    pts_L, pts_R, matches = match_features(img_L, img_R)

    print("<------- estimate right camera pose ------->")
    print(f"Number of detected matching features: {len(matches)}")

    # 2. Restore the cropped images
    pixel_L = (pts_L + crop_offset_L).astype(int)
    pixel_R = (pts_R + crop_offset_R).astype(int)

    ## 2.1 Mask pixels out of range
    mask_in_range = (
        pixel_L[:, 1] >= 0
        & (pixel_L[:, 1] < depth_L.shape[0])
        & (pixel_L[:, 0] >= 0)
        & (pixel_L[:, 0] < depth_L.shape[1])
    )

    pixel_L = pixel_L[mask_in_range]
    pixel_R = pixel_R[mask_in_range]

    print(f"Number of in-range matching features: {len(pixel_L)}\n")

    ## 2.2 plot the matching result
    img_L_np = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
    img_R_np = cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB)
    h1, w1 = img_L_np.shape[:2]
    h2, w2 = img_R_np.shape[:2]
    canvas_h = h1 + h2
    canvas_w = max(w1, w2)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img_L_np
    canvas[h1:h1+h2, :w2] = img_R_np

    num_matches = min(len(matches), 100)
    plt.figure(figsize=(8, 12))
    plt.title(f"Feature Matching Result, sampled {num_matches} matches")
    plt.imshow(canvas)
    plt.axis('off')

    cmap = plt.cm.viridis
    for i in range(0, num_matches):
        x1, y1 = pts_L[i]
        x2, y2 = pts_R[i]
        plt.plot([x1, x2], [y1, y2+h1], color=cmap(i/num_matches), linewidth=2)
    plt.show(block=visualize)

    # 3. Get the pointcloud of corresponding pixels
    pixel_mask = np.zeros_like(depth_L)
    pixel_mask[pixel_L[:, 1], pixel_L[:, 0]] = 1
    pcd, xs, ys = depth_to_pointcloud(depth_L, KL, pixel_mask)

    ## 3.1 Not all pixels can found corresponding 3D points, filter here
    tree = cKDTree(pixel_L)
    _, idx = tree.query(np.hstack([xs.reshape(-1, 1), ys.reshape(-1, 1)], dtype=np.int32), k=1)

    mask_matched = (idx >= 0)
    pcd_matched = pcd[mask_matched].astype(np.float32)
    idx_matched = idx[mask_matched]
    pixel_R_matched = pixel_R[idx_matched].astype(np.float32)
    
    print("<------- estimate right camera pose ------->")
    print(f"Number of valid (pixel, 3D points) pairs: {len(pcd_matched)}\n")

    # 4. Use PnP to estimate the right camera pose
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        pcd_matched,
        pixel_R_matched,
        KR,
        DR,
        flags=cv2.SOLVEPNP_ITERATIVE,
        reprojectionError=3.0,
        confidence=0.99
    )
    
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3,)

    return R, t

def task2(meta_file_path: str, single_shot_number: int = 0, visualize: bool = True):
    """
    Args:
        meta_file_path (str): Path to meta file.
        single_shot_number (int): Which single-shot image pair to use.
    """

    meta = read_meta(meta_file_path)

    # Load image and depth
    img_dir = os.path.join(os.path.dirname(meta_file_path), "single_shot")
    img_L = cv2.imread(os.path.join(img_dir, f"cam_L_{single_shot_number:04d}.png"))
    img_R = cv2.imread(os.path.join(img_dir, f"cam_R_{single_shot_number:04d}.png"))

    depth_L = np.load(os.path.join(os.path.dirname(meta_file_path), "depth_alacarte32.npy"))

    # Heuristically set the crop size
    h, w = img_L.shape[:2]
    y0, y1, x0, x1 = h // 4, 2 * h // 3, 1 * w // 5, 5 * w // 6
    img_L_cropped = img_L[y0:y1, x0:x1]
    img_R_cropped = img_R[y0:y1, x0:x1]

    R, t = estimate_right_camera_pose(
        meta["KL"],
        meta["KR"],
        meta["DR"],
        depth_L,
        img_L_cropped,
        img_R_cropped,
        np.array([x0, y0]),
        np.array([x0, y0]),
        visualize=visualize
    )

    print("<------- pose estimation result ------->")
    print(f"projector: \nR:\n {meta['R']}\nt:\n {meta['T'].flatten()}\n")
    print(f"right camera: \nR:\n {R}\nt:\n {t}")
    cam_R_pose = np.eye(4, dtype=np.float32)
    cam_R_pose[:3, :3] = R
    cam_R_pose[:3, 3] = t.flatten() / 1000.0  # mm to meters
    np.save(os.path.join(os.path.dirname(meta_file_path), "cam_R_pose.npy"), cam_R_pose)

if __name__ == '__main__':
    meta_file_path = 'data/objects/book/meta.npy'
    task2(meta_file_path, single_shot_number=0, visualize=True)