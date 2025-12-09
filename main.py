from paint_by_numbers import PaintByNumbers
import time

def main():
    image_path = "images/starry-night.jpg" # Make sure this image exists
    
    print(f"Loading {image_path}...")
    pbn = PaintByNumbers(image_path)
    
    start_time = time.perf_counter()
    
    (
        pbn.reset()
        .saturate(scale_factor=1.5)
        .set_palette(n_clusters=16)
        .recolor_with_palette()
        .apply_brush(radius=3)
        .remove_small_regions(min_area=250)
        .remove_thin_regions(min_thickness=3)
        .refine_segments(kernel_size=9, min_area=250)
        .draw_shared_borders(smoothing_size=11, padding=50)
        .generate_labels()
        .save_output("images/result")
    )
    
    end_time = time.perf_counter()
    print(f"Done in {end_time - start_time:.2f} seconds.")
    
    # pbn.display_results()

if __name__ == "__main__":
    main()