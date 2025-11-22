import numpy as np
import cv2

def to_gray(img: np.ndarray):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        return img

def build_multichannel_descriptor(img_list: list):
    return np.stack(img_list, axis=2).astype(np.float32)

def adjust_intrinsics_after_crop(P: np.ndarray, crop_x: int, crop_y: int):
    """Adjust intrinsics matrix after cropping the image.

    Args:
        P (np.ndarray): 3x4 projection matrix.
        crop_x (int): Crop width.
        crop_y (int): Crop height.

    Returns:
        np.ndarray: Adjusted projection matrix.
    """

    P2 = P.copy()
    P2[0, 2] -= crop_x
    P2[1, 2] -= crop_y
    return P

def rectify_and_crop(
    meta: dict, 
    left_img_list: list[np.ndarray], 
    right_img_list: list[np.ndarray], 
    alpha=0.0
):
    h, w = left_img_list[0].shape[:2]
    KL, DL, KP, DP, R, T = meta['KL'], meta['DL'], meta['KP'], meta['DP'], meta['R'], meta['T']

    # stereoRectify
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        KL, DL, KP, DP, (w, h), R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha)

    # maps
    map1x, map1y = cv2.initUndistortRectifyMap(KL, DL, R1, P1, (w, h), cv2.CV_32FC1)
    map2x, map2y = cv2.initUndistortRectifyMap(KP, DP, R2, P2, (w, h), cv2.CV_32FC1)

    left_rect = [cv2.remap(img, map1x, map1y, interpolation=cv2.INTER_LINEAR) for img in left_img_list]
    right_rect = [cv2.remap(img, map2x, map2y, interpolation=cv2.INTER_LINEAR) for img in right_img_list]

    # convert to gray
    left_gray = [to_gray(img) for img in left_rect]
    left_gray[-1] = left_rect[-1] ## original RGB, channel = 3
    right_gray = [to_gray(img) for img in right_rect]
    mask = [(p > 0).astype(np.uint8) for p in right_gray]
    mask_stk = np.stack(mask, axis=2)
    mask_gray = np.sum(mask_stk, axis=2) > 0 ## [h, w]

    # crop the image
    ys, xs = np.where(mask_gray)
    if len(xs) == 0:
        raise ValueError("No valid projection region found.")
    x0, x1, y0, y1 = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
    
    print("<------- rectify and crop ------->\n"
          f"Original image size: ({w}, {h})\n"
          f"Crop region for image: ({x0}, {x1}), ({y0}, {y1})\n"
          f"KL: \n{KL}\nKP: \n{KP}\n"
          f"R1: \n{R1}\nR2: \n{R2}\nP1: \n{P1}\nP2: \n{P2}\nQ: \n{Q}\n")

    left_crop = [img[y0:y1+1, x0:x1+1] for img in left_gray]
    right_crop = [img[y0:y1+1, x0:x1+1] for img in right_gray]  
    mask_crop = mask_gray[y0:y1+1, x0:x1+1]

    # adjust intrinsics
    P1_crop = adjust_intrinsics_after_crop(P1, x0, y0)
    P2_crop = adjust_intrinsics_after_crop(P2, x0, y0)

    return left_crop, right_crop, mask_crop, P1_crop, P2_crop, R1, R2, Q, (x0, y0)

def disp_to_depth(disp: np.ndarray, P: np.ndarray, baseline: float):
    """
    Convert disparity map to depth map.
    """

    f = P[0, 0]
    depth = np.zeros_like(disp, dtype=np.float32)
    valid = (disp > 0) & (~np.isnan(disp))
    depth[valid] = (f * abs(baseline)) / disp[valid]
    depth[depth > 6e3] = 0 # filter out depth>6m
    depth[~valid] = 0
    
    print(f"<------- disp to depth ------->\n"
          f"f: {f}, baseline: {baseline}, \n"
          f"disp.max: {np.nanmax(disp)}, disp.min: {np.nanmin(disp)}, \n"
          f"depth.max: {depth[valid].max()}, depth.min: {depth[valid].min()}\n")
    
    return depth


def depth_to_pointcloud(depth: np.ndarray, P: np.ndarray, mask: np.ndarray=None):
    """
    Convert depth map to point cloud.
    """

    fx, fy, cx, cy = P[0, 0], P[1, 1], P[0, 2], P[1, 2]

    ys, xs = np.where((depth > 0) if mask is None else (depth > 0) & (mask > 0))
    zs = depth[ys, xs]
    xs_cam = (xs - cx) * zs / fx
    ys_cam = (ys - cy) * zs / fy
    points = np.vstack([xs_cam, ys_cam, zs]).T
    print("<------- depth to pointcloud ------->\n"
          f"Number of valid points: {len(xs)}, \n"
          f"x.max: {xs_cam.max()}, x.min: {xs_cam.min()}, \n"
          f"y.max: {ys_cam.max()}, y.min: {ys_cam.min()}, \n"
          f"z.max: {zs.max()}, z.min: {zs.min()}, \n"
          f"cx: {cx}, cy: {cy}, fx: {fx}, fy: {fy}\n")

    return points, xs, ys
