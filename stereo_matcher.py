from multiprocessing import Pool, cpu_count

import numpy as np
from numba import njit, prange

##################################
##         Stereo Matcher       ##
##################################

def compute_row_disp_vectorized(Lrow: np.ndarray, Rrow: np.ndarray, mask_row: np.ndarray, max_disp: int, win: int):
    """
    Lrow, Rrow: (w, c) float32
    mask_row: (w,) uint8 (0/1). If None, all ones.
    returns disp_row: (w,) float32 with nan where invalid
    """
    w, c = Lrow.shape
    half = win // 2

    # Compute valid x indices
    xs = np.arange(half, w-half)

    # window sum for each center x is cs[x+half+1] - cs[x-half]
    cs_left = xs- half
    cs_right = xs + half + 1

    # Get mask
    if mask_row is None:
        mask_row = np.ones((w,), dtype=np.uint8)

    # Build masked L and R: (w, c)
    mask_col = mask_row.astype(Lrow.dtype)
    L_masked = Lrow * mask_col[:,None]
    R_masked = Rrow * mask_col[:,None]

    # Precompute channel sums and squares for L and L2 (for all window centers)
    csLm = np.vstack((np.zeros((1, c), dtype=Lrow.dtype), np.cumsum(L_masked, axis=0)))
    csL2m = np.vstack((np.zeros((1, c), dtype=Lrow.dtype), np.cumsum(L_masked * L_masked, axis=0)))
    # per-center sums:
    sumLm_win = csLm[cs_right, :] - csLm[cs_left, :]
    sumL2m_win = csL2m[cs_right, :] - csL2m[cs_left, :]

    # Count valid pixels in each centered window
    csMask = np.concatenate((np.zeros((1,), dtype=np.int32), np.cumsum(mask_row.astype(np.int32))), axis=0)
    count_win = (csMask[cs_right] - csMask[cs_left]) * c  # shape (n_centers,)

    # Now iterate disparities: for each d, shift R_masked by d (right->left shift) and compute:
    # - sumR_win, sumR2_win (per center and per channel)
    # - cross = sum over window and channels of L_masked * R_shifted_masked
    n_centers = len(xs)
    best_costs = np.full(n_centers, -1e9, dtype=np.float32)
    best_disp = np.zeros(n_centers, dtype=np.float32)

    max_d = min(max_disp, w)  # safe upper bound
    for d in range(1, max_d):
        valid_idx = np.where(count_win > 0)[0]
        if valid_idx.size == 0:
            continue

        # shift R_masked to align with L: R_shift[x] = R_masked[x - d]
        R_shift = np.zeros_like(R_masked)
        R_shift[d:, :] = R_masked[:-d, :]

        # compute cs for R_shift
        csR = np.vstack((np.zeros((1, c), dtype=R_shift.dtype), np.cumsum(R_shift, axis=0)))
        csR2 = np.vstack((np.zeros((1, c), dtype=R_shift.dtype), np.cumsum(R_shift * R_shift, axis=0)))

        sumRm_win = csR[cs_right, :] - csR[cs_left, :]
        sumR2m_win = csR2[cs_right, :] - csR2[cs_left, :]

        # cross term: we need sum over window and channels of (L_masked * R_shift)
        # compute elementwise product then box sum across x
        prod = L_masked * R_shift  # (w, c)
        csProd = np.vstack((np.zeros((1, c), dtype=prod.dtype), np.cumsum(prod, axis=0)))
        sumProd_win = csProd[cs_right, :] - csProd[cs_left, :]  # shape (n_centers, c)
        # sum over channels
        sumProd = np.sum(sumProd_win, axis=1)[valid_idx]  # shape (n_centers,)

        # sums over channels for L and R
        sL = np.sum(sumLm_win, axis=1)[valid_idx]   # shape (len(valid_index),)
        sR = np.sum(sumRm_win, axis=1)[valid_idx]
        sL2 = np.sum(sumL2m_win, axis=1)[valid_idx]
        sR2 = np.sum(sumR2m_win, axis=1)[valid_idx]

        # Now compute NCC per center: ncc = (sumProd - count * meanL * meanR) / sqrt((sL2 - count*meanL^2)*(sR2 - count*meanR^2))
        # But meanL = sL / count, meanR = sR / count
        # Numerator = sumProd - sL * sR / count
        # Denominator = sqrt((sL2 - sL^2 / count)*(sR2 - sR^2 / count))
        # Avoid divisions by zero: require count>0 and denom>epsilon
        eps = 1e-8

        cnt = count_win[valid_idx].astype(np.float32)
        numer = sumProd - (sL * sR) / cnt
        denom_left = sL2 - (sL * sL) / cnt
        denom_right = sR2 - (sR * sR) / cnt
        denom = denom_left * denom_right
        
        # handle small/negative denom due to numeric issues
        mask_denom = denom < eps
        denom = np.where(mask_denom, np.inf, denom)
        mask_invalid_center = (xs < (d + half))

        ncc = np.full(n_centers, -1e8, dtype=np.float32)
        ncc[valid_idx] = numer / np.sqrt(denom)
        ncc[valid_idx][mask_denom] = -1e8  # invalid
        ncc[mask_invalid_center] = -1e8    # invalid

        # update best
        better = ncc > best_costs
        if np.any(better):
            best_costs[better] = ncc[better]
            best_disp[better] = d

    # scatter best_disp back to full width, put NaN for invalid x
    disp_row = np.full(w, np.nan, dtype=np.float32)
    disp_row[xs] = best_disp
    return disp_row

def _worker_unpack_args(args):
    return compute_row_disp_vectorized(*args)

def match_multichannel_rowwise_fast(left_desc, right_desc, mask=None, max_disp=128, win=5, num_workers=None):
    """
    Vectorized per-row matching, parallelized over rows.
    left_desc, right_desc: (H, W, C) float32
    mask: (H, W) uint8 or None
    """
    H, W, C = left_desc.shape
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)

    args = []
    for y in range(H):
        Lrow = left_desc[y].astype(np.float32)
        Rrow = right_desc[y].astype(np.float32)
        mask_row = None if mask is None else mask[y].astype(np.uint8)
        args.append((Lrow, Rrow, mask_row, max_disp, win))

    # Use multiprocessing Pool
    with Pool(processes=num_workers) as p:
        results = p.map(_worker_unpack_args, args)

    disp = np.stack(results, axis=0)
    return disp
