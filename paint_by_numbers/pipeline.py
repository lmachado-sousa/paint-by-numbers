import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from typing import Tuple

# Import modules
from . import processing
from . import rendering

class PaintByNumbers:
    """A pipeline for creating paint-by-numbers artwork from images.
    
    This class provides a complete workflow for transforming photographs into
    paint-by-numbers style artwork with numbered regions and color palettes.
    The pipeline includes image preprocessing, color quantization, region
    refinement, and output generation with borders and labels.
    
    Attributes:
        original_image (np.ndarray): The original input image in RGB format.
        transformed_image (np.ndarray): The current state of the processed image.
        palette (np.ndarray): Array of color values representing the color palette.
        n_clusters (int): Number of colors in the palette.
        height (int): Current height of the transformed image.
        width (int): Current width of the transformed image.
    
    Example:
        >>> pbn = PaintByNumbers("image.jpg")
        >>> pbn.resize(2.0).saturate().set_palette(16)
        >>> pbn.recolor_with_palette().apply_brush()
        >>> pbn.draw_shared_borders().generate_labels()
        >>> pbn.save_output("my_painting")
    """
    def __init__(self, image_path: str) -> None:
        """Initialize a PaintByNumbers instance with an image.
        
        Args:
            image_path (str): Path to the input image file.
        """
        self.original_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.transformed_image = self.original_image.copy()
        self.palette = None
        self.n_clusters = 0
        self._update_dimensions()

    def _update_dimensions(self) -> None:
        """Update the stored height and width from the current transformed image."""
        self.height, self.width = self.transformed_image.shape[:2]

    def _assert_palette_exists(self) -> None:
        """Verify that a color palette has been set.
        
        Raises:
            RuntimeError: If palette has not been initialized via set_palette().
        """
        if self.palette is None:
            raise RuntimeError("Palette not set. Call set_palette() first.")

    def reset(self) -> 'PaintByNumbers':
        """Reset the transformed image to the original image.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        """
        self.transformed_image = self.original_image.copy()
        self._update_dimensions()
        return self

    def resize(self, factor: float = 1.0) -> 'PaintByNumbers':
        """Resize the image by dividing dimensions by the given factor.
        
        Args:
            factor (float): Divisor for width and height. Values > 1 reduce size,
                          values < 1 increase size. Defaults to 1.0 (no change).
        
        Returns:
            PaintByNumbers: Self for method chaining.
        """
        new_w, new_h = int(self.width / factor), int(self.height / factor)
        self.transformed_image = cv2.resize(self.transformed_image, (new_w, new_h))
        self._update_dimensions()
        return self

    def saturate(self, scale_factor: float = 2.5) -> 'PaintByNumbers':
        """Increase the color saturation of the image.
        
        Args:
            scale_factor (float): Saturation multiplier. Values > 1 increase
                                saturation. Defaults to 2.5.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        """
        self.transformed_image = processing.saturate_image(self.transformed_image, scale_factor)
        return self

    def set_palette(self, n_clusters: int = 16) -> 'PaintByNumbers':
        """Generate a color palette by quantizing the image colors.
        
        Args:
            n_clusters (int): Number of distinct colors in the palette.
                            Defaults to 16.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        """
        self.palette, _ = processing.quantize_image(self.transformed_image, n_clusters)
        self.n_clusters = n_clusters
        self.palette_tree = KDTree(self.palette) # Cache tree for matching
        return self

    def recolor_with_palette(self) -> 'PaintByNumbers':
        """Replace all pixels with their nearest color from the palette.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        
        Raises:
            RuntimeError: If palette has not been set.
        """
        self._assert_palette_exists()
        h, w = self.height, self.width
        
        # Match pixels to palette using KDTree
        pixels = self.transformed_image.reshape(-1, 3)
        _, indices = self.palette_tree.query(pixels)
        self.transformed_image = self.palette[indices].reshape(h, w, 3).astype(np.uint8)
        
        return self

    def apply_brush(self, radius: int = None) -> 'PaintByNumbers':
        """Apply a brush-like smoothing effect to the image.
        
        Args:
            radius (int, optional): Brush radius in pixels. If None, an optimal
                                   radius is calculated based on image dimensions.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        
        Raises:
            RuntimeError: If palette has not been set.
        """
        self._assert_palette_exists()
        if radius is None:
            optimal = np.round(np.sqrt(7e-6 * self.width * self.height))
            radius = max(int(optimal), 4)
            
        # Apply filter
        blurred = processing.apply_brush_filter(self.transformed_image, radius)
        
        # Snap back to palette
        _, indices = self.palette_tree.query(blurred.reshape(-1, 3))
        self.transformed_image = self.palette[indices].reshape(self.height, self.width, 3).astype(np.uint8)
        
        return self

    def remove_small_regions(self, min_area: int = 500) -> 'PaintByNumbers':
        """Remove small color regions by merging them into neighboring regions.
        
        Args:
            min_area (int): Minimum region size in pixels. Regions smaller than
                          this will be removed. Defaults to 500.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        
        Raises:
            RuntimeError: If palette has not been set.
        """
        self._assert_palette_exists()
        self.transformed_image = processing.remove_small_regions_logic(
            self.transformed_image, self.palette, min_area
        )
        return self

    def remove_thin_regions(self, min_thickness: int = 3) -> 'PaintByNumbers':
        """Remove thin color regions that are difficult to paint.
        
        Args:
            min_thickness (int): Minimum region thickness in pixels. Thinner
                               regions will be removed. Defaults to 3.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        
        Raises:
            RuntimeError: If palette has not been set.
        """
        self._assert_palette_exists()
        self.transformed_image = processing.remove_thin_regions_logic(
            self.transformed_image, self.palette, min_thickness
        )
        return self

    def refine_segments(self, kernel_size: int = 7, min_area: int = 500) -> 'PaintByNumbers':
        """Refine color segments using morphological operations.
        
        Args:
            kernel_size (int): Size of morphological kernel. Defaults to 7.
            min_area (int): Minimum region size to keep in pixels. Defaults to 500.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        
        Raises:
            RuntimeError: If palette has not been set.
        """
        self._assert_palette_exists()
        self.transformed_image = processing.refine_segments_logic(
            self.transformed_image, self.palette, kernel_size, min_area
        )
        return self

    def draw_shared_borders(self, line_thickness: int = 1, smoothing_size: int = 3, padding: int = 50) -> 'PaintByNumbers':
        """Generate bordered canvas and preview images with color legend.
        
        Args:
            line_thickness (int): Thickness of border lines in pixels. Defaults to 1.
            smoothing_size (int): Size of smoothing kernel for border detection.
                                Defaults to 3.
            padding (int): White border padding around images in pixels. Defaults to 50.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        
        Raises:
            RuntimeError: If palette has not been set.
        """
        self._assert_palette_exists()
        h, w = self.height, self.width
        self.output_padding = padding

        # 1. Detect Borders
        edge_mask = rendering.generate_borders(
            self.transformed_image, self.palette, line_thickness, smoothing_size
        )

        # 2. Create Base Images
        base_canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        base_canvas[edge_mask] = (180, 180, 180) # Light Gray Lines
        
        base_preview = self.transformed_image.copy()
        base_preview[edge_mask] = (60, 60, 60)   # Dark Gray Lines

        # 3. Add Padding & Frames
        palette_height = 150
        
        # Manual Padding logic here to account for palette space on canvas only
        self.canvas = cv2.copyMakeBorder(
            base_canvas, padding, padding + palette_height, padding, padding,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        self.preview_image = cv2.copyMakeBorder(
            base_preview, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        
        # Draw Frames
        cv2.rectangle(self.canvas, (padding, padding), (w + padding, h + padding), (0, 0, 0), 2)
        cv2.rectangle(self.preview_image, (padding, padding), (w + padding, h + padding), (0, 0, 0), 2)

        # 4. Draw Legend
        self.canvas = rendering.draw_palette_legend(self.canvas, self.palette, padding, h + padding + 20, w)
        
        return self

    def generate_labels(self, min_font_scale: float = 0.4, max_font_scale: float = 1.0) -> 'PaintByNumbers':
        """Add numbered labels to each color region on the canvas.
        
        Args:
            min_font_scale (float): Minimum font size scale. Defaults to 0.4.
            max_font_scale (float): Maximum font size scale. Defaults to 1.0.
        
        Returns:
            PaintByNumbers: Self for method chaining.
        """
        if not hasattr(self, 'canvas'):
            self.draw_shared_borders()
            
        self.canvas, self.preview_image = rendering.draw_labels(
            self.canvas, self.preview_image, self.transformed_image, self.palette,
            self.output_padding, min_font_scale, max_font_scale
        )
        return self

    def display_results(self, figsize: Tuple[int, int] = (16, 8)) -> None:
        """Display the canvas and preview images side by side.
        
        Args:
            figsize (Tuple[int, int]): Figure size as (width, height) in inches.
                                      Defaults to (16, 8).
        """
        if not hasattr(self, 'canvas'):
            print("Run generate_labels() first.")
            return

        fig, axs = plt.subplots(1, 2, figsize=figsize)
        axs[0].imshow(self.canvas)
        axs[0].set_title("Printable Canvas")
        axs[0].axis('off')
        
        axs[1].imshow(self.preview_image)
        axs[1].set_title("Reference Preview")
        axs[1].axis('off')
        
        plt.tight_layout()
        plt.show()

    def save_output(self, prefix: str = "output") -> None:
        """Save the canvas and preview images to PNG files.
        
        Args:
            prefix (str): Filename prefix for output files. Files will be saved as
                        '{prefix}_canvas.png' and '{prefix}_preview.png'.
                        Defaults to "output".
        """
        if not hasattr(self, 'canvas'):
            print("Please run generate_labels() first.")
            return

        canvas_bgr = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)
        preview_bgr = cv2.cvtColor(self.preview_image, cv2.COLOR_RGB2BGR)
        
        cv2.imwrite(f"{prefix}_canvas.png", canvas_bgr)
        cv2.imwrite(f"{prefix}_preview.png", preview_bgr)
        print(f"Saved: {prefix}_canvas.png & {prefix}_preview.png")