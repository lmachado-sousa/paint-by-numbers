"""Image processing functions for paint-by-numbers generation.

This module contains pure, stateless functions for image manipulation,
color quantization, and morphological operations. All functions follow
the pattern: Image In -> Image Out.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

def saturate_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    """Increase the color saturation of an image.
    
    Args:
        image (np.ndarray): Input image in RGB format.
        scale_factor (float): Saturation multiplier. Values > 1 increase saturation.
    
    Returns:
        np.ndarray: Image with adjusted saturation in RGB format.
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv_image[:, :, 1] *= scale_factor
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2RGB)

def quantize_image(image: np.ndarray, n_clusters: int) -> tuple[np.ndarray, np.ndarray]:
    """Reduce image colors using K-Means clustering in Lab color space.
    
    Performs color quantization by clustering pixels in perceptually uniform
    Lab color space, then converts cluster centers back to RGB.
    
    Args:
        image (np.ndarray): Input image in RGB format.
        n_clusters (int): Number of distinct colors to generate.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - palette_rgb: Array of shape (n_clusters, 3) with RGB color values.
            - labels: Cluster labels for the resized image (for memory efficiency).
    """
    h, w = image.shape[:2]
    
    # Resize for speed optimization during clustering
    ideal_resize = max(w, h) / 250
    small_w, small_h = int(w / ideal_resize), int(h / ideal_resize)
    small_image = cv2.resize(image, (small_w, small_h))

    # Convert to Lab
    pixels = cv2.cvtColor(small_image.astype(np.float32) / 255.0, cv2.COLOR_RGB2Lab).reshape(-1, 3)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(pixels)

    # Convert centers back to RGB
    palette_lab = kmeans.cluster_centers_.astype(np.float32)
    palette_rgb = cv2.cvtColor(palette_lab.reshape(1, -1, 3), cv2.COLOR_Lab2RGB).reshape(-1, 3) * 255
    
    return palette_rgb.astype(int), kmeans.labels_

def apply_brush_filter(image: np.ndarray, radius: int) -> np.ndarray:
    """Apply a circular averaging filter to smooth the image.
    
    Creates a brush-like smoothing effect using a circular kernel convolution.
    
    Args:
        image (np.ndarray): Input image in RGB format.
        radius (int): Radius of the circular smoothing kernel in pixels.
    
    Returns:
        np.ndarray: Smoothed image as float32 array.
    """
    # Create circular kernel
    y, x = np.ogrid[-radius:radius+1, -radius:radius+1]
    mask = x**2 + y**2 <= radius**2
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
    kernel[mask] = 1
    kernel /= mask.sum()

    mean_colors = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        mean_colors[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel, borderType=cv2.BORDER_REFLECT)
    
    return mean_colors

def _fill_holes(id_map: np.ndarray, hole_value: int = -1) -> np.ndarray:
    """Fill holes in an ID map by propagating neighboring values.
    
    Iteratively dilates valid region IDs into areas marked with hole_value
    until all holes are filled.
    
    Args:
        id_map (np.ndarray): 2D array of region IDs.
        hole_value (int): Value marking holes to be filled. Defaults to -1.
    
    Returns:
        np.ndarray: ID map with all holes filled.
    """
    fixed_map = id_map.copy()
    mask_bad = (fixed_map == hole_value)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while np.any(mask_bad):
        # Dilate valid IDs into the holes
        # Casting to float32 is required for cv2.dilate if using negative numbers
        dilated = cv2.dilate(fixed_map.astype(np.float32), kernel).astype(np.int32)
        fixed_map[mask_bad] = dilated[mask_bad]
        mask_bad = (fixed_map == hole_value)
    
    return fixed_map

def remove_small_regions_logic(image: np.ndarray, palette: np.ndarray, min_area: int) -> np.ndarray:
    """Remove small disconnected color regions from the image.
    
    Identifies connected components for each color and removes those smaller
    than the minimum area threshold. Removed regions are filled by propagating
    neighboring color values.
    
    Args:
        image (np.ndarray): Input image in RGB format with palette colors.
        palette (np.ndarray): Array of RGB color values used in the image.
        min_area (int): Minimum region size in pixels. Smaller regions are removed.
    
    Returns:
        np.ndarray: Image with small regions removed and filled.
    """
    h, w = image.shape[:2]
    n_clusters = len(palette)
    
    # 1. Generate ID map
    id_map = np.zeros((h, w), dtype=np.int32)
    for i, color in enumerate(palette):
        mask = cv2.inRange(image, color, color)
        id_map[mask > 0] = i

    # 2. Find Small Regions
    to_remove = np.zeros((h, w), dtype=np.uint8)
    for i in range(n_clusters):
        c_mask = (id_map == i).astype(np.uint8)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(c_mask, connectivity=4)
        
        # stats[:, 4] is Area
        small_idxs = np.where(stats[:, cv2.CC_STAT_AREA] < min_area)[0]
        small_idxs = small_idxs[small_idxs > 0] # Skip background
        
        if len(small_idxs) > 0:
            to_remove[np.isin(labels, small_idxs)] = 255

    # 3. Fill
    id_map[to_remove == 255] = -1
    fixed_id_map = _fill_holes(id_map, -1)
    
    # 4. Reconstruct
    return palette[fixed_id_map].astype(np.uint8)

def remove_thin_regions_logic(image: np.ndarray, palette: np.ndarray, min_thickness: int) -> np.ndarray:
    """Remove thin color regions using morphological opening.
    
    Applies morphological opening (erosion followed by dilation) to each color
    region to eliminate thin structures that would be difficult to paint.
    
    Args:
        image (np.ndarray): Input image in RGB format with palette colors.
        palette (np.ndarray): Array of RGB color values used in the image.
        min_thickness (int): Minimum thickness in pixels. Thinner regions are removed.
    
    Returns:
        np.ndarray: Image with thin regions removed and filled.
    """
    h, w = image.shape[:2]
    id_map = np.zeros((h, w), dtype=np.int32)
    for i, color in enumerate(palette):
        mask = cv2.inRange(image, color, color)
        id_map[mask > 0] = i

    kernel_size = max(3, min_thickness)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    bad_pixels = np.zeros((h, w), dtype=bool)
    
    for i in range(len(palette)):
        c_mask = (id_map == i).astype(np.uint8) * 255
        # Open = Erode -> Dilate
        clean = cv2.morphologyEx(c_mask, cv2.MORPH_OPEN, kernel)
        lost = (c_mask > 0) & (clean == 0)
        bad_pixels[lost] = True

    id_map[bad_pixels] = -1
    fixed_id_map = _fill_holes(id_map, -1)
    
    return palette[fixed_id_map].astype(np.uint8)

def refine_segments_logic(image: np.ndarray, palette: np.ndarray, kernel_size: int, min_area_cleanup: int) -> np.ndarray:
    """Refine color segments using median filtering and artifact removal.
    
    Applies median blur to smooth region boundaries and then removes small
    artifact regions created during the smoothing process.
    
    Args:
        image (np.ndarray): Input image in RGB format with palette colors.
        palette (np.ndarray): Array of RGB color values used in the image.
        kernel_size (int): Size of median blur kernel. Should be odd.
        min_area_cleanup (int): Minimum region size to keep after smoothing.
    
    Returns:
        np.ndarray: Image with refined and cleaned segments.
    """
    h, w = image.shape[:2]
    
    # 1. ID Map
    id_map = np.zeros((h, w), dtype=np.int32)
    for i, color in enumerate(palette):
        mask = cv2.inRange(image, color, color)
        id_map[mask > 0] = i

    # 2. Median Blur
    if kernel_size > 1:
        if kernel_size % 2 == 0: kernel_size += 1
        blurred = cv2.medianBlur(id_map.astype(np.uint8), kernel_size)
        id_map = blurred.astype(np.int32)

    # 3. Cleanup Specks
    to_remove = np.zeros((h, w), dtype=np.uint8)
    for i in range(len(palette)):
        c_mask = (id_map == i).astype(np.uint8)
        if cv2.countNonZero(c_mask) == 0: continue
        
        _, labels, stats, _ = cv2.connectedComponentsWithStats(c_mask, connectivity=4)
        small = np.where(stats[:, cv2.CC_STAT_AREA] < min_area_cleanup)[0]
        small = small[small > 0]
        if len(small) > 0:
            to_remove[np.isin(labels, small)] = 255
            
    id_map[to_remove == 255] = -1
    fixed_id_map = _fill_holes(id_map, -1)
    
    return palette[fixed_id_map].astype(np.uint8)