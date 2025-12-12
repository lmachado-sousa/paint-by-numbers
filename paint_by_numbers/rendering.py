"""Rendering functions for paint-by-numbers visualization.

This module contains functions for drawing borders, creating layouts,
placing text labels, and generating the final output images.
"""

import cv2
import numpy as np

def generate_borders(image: np.ndarray, palette: np.ndarray, 
                    line_thickness: int = 1, smoothing_size: int = 3) -> np.ndarray:
    """Detect borders between color regions using morphological gradients.
    
    Creates a binary mask indicating boundaries between different colored regions
    by applying morphological gradient operations to the color-mapped image.
    
    Args:
        image (np.ndarray): Input image in RGB format with palette colors.
        palette (np.ndarray): Array of RGB color values used in the image.
        line_thickness (int): Thickness of border lines in pixels. Defaults to 1.
        smoothing_size (int): Size of median blur kernel for smoothing before
                            edge detection. Must be odd. Defaults to 3.
    
    Returns:
        np.ndarray: Boolean mask where True indicates border pixels.
    """
    h, w = image.shape[:2]
    
    # Generate ID map
    id_map = np.zeros((h, w), dtype=np.uint8)
    for i, color in enumerate(palette):
        mask = cv2.inRange(image, color, color)
        id_map[mask > 0] = i

    # Morph Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.morphologyEx(id_map, cv2.MORPH_GRADIENT, kernel)

    if line_thickness > 1:
        dil_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (line_thickness, line_thickness))
        edges = cv2.dilate(edges, dil_kernel)

    return edges > 0

def create_padded_canvas(base_image: np.ndarray, padding: int, palette_height: int = 0) -> np.ndarray:
    """Add white border and black frame around an image.
    
    Wraps the input image with a white border and draws a black bounding box
    around the original image area. Optionally adds extra space at the bottom
    for a color palette legend.
    
    Args:
        base_image (np.ndarray): Input image to wrap.
        padding (int): Width of white border in pixels.
        palette_height (int): Extra space to add at bottom for palette legend.
                            Defaults to 0.
    
    Returns:
        np.ndarray: Padded image with border and frame.
    """
    h, w = base_image.shape[:2]
    
    # Add border (White)
    canvas = cv2.copyMakeBorder(
        base_image, 
        padding, 
        padding + palette_height if palette_height > 0 else padding, 
        padding, 
        padding, 
        cv2.BORDER_CONSTANT, value=(255, 255, 255)
    )
    
    # Draw Frame (Black)
    cv2.rectangle(canvas, (padding, padding), (w + padding, h + padding), (0, 0, 0), 2)
    
    return canvas

def draw_palette_legend(canvas: np.ndarray, palette: np.ndarray, 
                       start_x: int, start_y: int, width: int) -> np.ndarray:
    """Draw numbered color swatches at the bottom of the canvas.
    
    Creates a legend showing all palette colors as colored squares with their
    corresponding numeric labels. Automatically wraps to multiple rows if needed.
    
    Args:
        canvas (np.ndarray): Canvas image to draw on.
        palette (np.ndarray): Array of RGB color values to display.
        start_x (int): X coordinate for the left edge of the legend.
        start_y (int): Y coordinate for the top edge of the legend.
        width (int): Maximum width for the legend before wrapping.
    
    Returns:
        np.ndarray: Canvas with legend drawn.
    """
    swatch_size = 40
    gap = 10
    swatch_total_width = swatch_size + gap
    
    current_x = start_x
    current_y = start_y
    
    for i, color in enumerate(palette):
        # Row wrap
        if current_x + swatch_total_width > start_x + width:
            current_x = start_x
            current_y += swatch_size + 30
        
        pt1 = (current_x, current_y)
        pt2 = (current_x + swatch_size, current_y + swatch_size)
        c = (int(color[0]), int(color[1]), int(color[2]))
        
        cv2.rectangle(canvas, pt1, pt2, c, -1)
        cv2.rectangle(canvas, pt1, pt2, (0, 0, 0), 1)
        
        text = str(i)
        cv2.putText(canvas, text, (current_x + 5, current_y + swatch_size + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
        
        current_x += swatch_total_width
        
    return canvas

def draw_labels(canvas: np.ndarray, preview: np.ndarray, 
               image: np.ndarray, palette: np.ndarray, 
               padding: int, min_scale: float, max_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Add numbered labels to color regions on both canvas and preview images.
    
    Identifies all connected color regions, calculates optimal label positions
    using distance transform for visual centering, and draws appropriately
    scaled numeric labels on both output images.
    
    Args:
        canvas (np.ndarray): Canvas image (white with gray borders) to label.
        preview (np.ndarray): Preview image (colored with dark borders) to label.
        image (np.ndarray): Source image with palette colors for region detection.
        palette (np.ndarray): Array of RGB color values used in the image.
        padding (int): Border padding to account for when placing labels.
        min_scale (float): Minimum font scale for labels.
        max_scale (float): Maximum font scale for labels.
    
    Returns:
        tuple[np.ndarray, np.ndarray]: Canvas and preview images with labels drawn.
    """
    n_clusters = len(palette)
    label_locations = []

    # 1. Calculate Locations
    for i in range(n_clusters):
        color = palette[i]
        mask = cv2.inRange(image, color, color)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        
        for j in range(1, num):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < 10: continue

            # Distance Transform for visual center
            comp_mask = (labels == j).astype(np.uint8)
            dist = cv2.distanceTransform(comp_mask, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist)
            
            label_locations.append({
                "id": i, "center": max_loc, "area": area, "radius": max_val
            })

    # 2. Draw
    for item in label_locations:
        text = str(item["id"])
        
        # Scale font
        target_h = item["radius"]
        scale = max(min_scale, min(max_scale, target_h / 20.0))
        thickness = 1
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        
        x = int(item["center"][0] - tw / 2) + padding
        y = int(item["center"][1] + th / 2) + padding
        
        # Canvas: Dark Gray
        cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (80, 80, 80), thickness, cv2.LINE_AA)
        
        # Preview: White with Outline
        cv2.putText(preview, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(preview, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return canvas, preview