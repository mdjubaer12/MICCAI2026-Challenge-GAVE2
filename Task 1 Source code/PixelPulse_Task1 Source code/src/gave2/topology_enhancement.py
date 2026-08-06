"""Skeleton-guided topology enhancement and gap connection for GAVE2 probability maps."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import draw, morphology


def enhance_vessel_channel(
    channel_prob: np.ndarray,
    roi: np.ndarray,
    threshold: float = 0.35,
    max_gap_distance: float = 6.0,
    boost_val: float = 0.65,
) -> np.ndarray:
    """Connect broken skeleton endpoints and boost centerline probabilities above 0.50."""

    prob = np.asarray(channel_prob, dtype=float)
    roi_mask = np.asarray(roi, dtype=bool)
    bin_mask = (prob >= threshold) & roi_mask

    if not np.any(bin_mask):
        return (prob * 255.0).astype(np.uint8) if prob.max() <= 1.0 else prob.astype(np.uint8)

    # Skeletonize
    skel = morphology.skeletonize(bin_mask)
    if not np.any(skel):
        return (prob * 255.0).astype(np.uint8) if prob.max() <= 1.0 else prob.astype(np.uint8)

    # Find endpoints (skeleton pixels with 1 neighbor in 3x3 footprint)
    kernel = np.ones((3, 3), dtype=int)
    neighbors = ndimage.convolve(skel.astype(int), kernel, mode="constant") - skel.astype(int)
    endpoints = np.argwhere(skel & (neighbors == 1))

    enhanced_prob = prob.copy()

    # Pairwise endpoint distance search
    num_endpoints = len(endpoints)
    if num_endpoints > 1:
        for i in range(num_endpoints):
            pt1 = endpoints[i]
            for j in range(i + 1, num_endpoints):
                pt2 = endpoints[j]
                dist = float(np.linalg.norm(pt1 - pt2))
                if 1.5 <= dist <= max_gap_distance:
                    # Draw line connecting endpoints
                    rr, cc = draw.line(pt1[0], pt1[1], pt2[0], pt2[1])
                    # Mask within ROI
                    valid = (rr >= 0) & (rr < prob.shape[0]) & (cc >= 0) & (cc < prob.shape[1])
                    rr_v, cc_v = rr[valid], cc[valid]
                    valid_roi = roi_mask[rr_v, cc_v]
                    rr_v, cc_v = rr_v[valid_roi], cc_v[valid_roi]

                    # Boost probabilities along the connecting path
                    enhanced_prob[rr_v, cc_v] = np.maximum(enhanced_prob[rr_v, cc_v], boost_val)

    # Mild centerline probability boosting
    enhanced_prob[skel & roi_mask] = np.maximum(enhanced_prob[skel & roi_mask], boost_val)

    # Convert to uint8 [0, 255]
    if enhanced_prob.max() <= 1.0:
        return np.clip(np.rint(enhanced_prob * 255.0), 0, 255).astype(np.uint8)
    return np.clip(np.rint(enhanced_prob), 0, 255).astype(np.uint8)


def enhance_av_probability_map(
    probabilities_rgb: np.ndarray,
    roi: np.ndarray,
    threshold: float = 0.35,
    max_gap_distance: float = 6.0,
) -> np.ndarray:
    """Enhance HxWx3 artery/vessel/vein probability map topology."""

    prob = np.asarray(probabilities_rgb, dtype=float)
    if prob.max() > 1.0:
        prob = prob / 255.0

    art_enh = enhance_vessel_channel(prob[..., 0], roi, threshold, max_gap_distance)
    vei_enh = enhance_vessel_channel(prob[..., 2], roi, threshold, max_gap_distance)
    ves_enh = np.maximum(art_enh, vei_enh)

    return np.stack((art_enh, ves_enh, vei_enh), axis=-1)
