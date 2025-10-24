import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
from ASN_detector import ASN_detector, ASNConfig

#copyright intellar@intellar.ca

def display_results(img: np.ndarray, pts: np.ndarray, accumulateur: np.ndarray, convergence_regions: np.ndarray):
    """Displays the original image with detected points and overlays."""
    if img.ndim == 2:
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        cimg = img.copy()

    # Normalize accumulator for visualization
    acc_viz = cv2.normalize(accumulateur, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # Create a colored overlay for the labeled convergence regions
    # We use a colormap to give each region a unique color.
    convergence_viz = np.zeros_like(cimg)
    if convergence_regions.max() > 0:
        # --- Fast Visualization using a Lookup Table (LUT) ---
        # 1. Get the number of unique labels.
        num_labels = convergence_regions.max() + 1
        
        # 2. Create a color lookup table.
        # We generate colors from a colormap for each label ID.
        lut = np.zeros((num_labels, 3), dtype=np.uint8)
        colors = (plt.cm.viridis(np.linspace(0, 1, num_labels))[:, :3] * 255).astype(np.uint8)
        lut[1:] = colors[1:] # Assign colors, keeping label 0 (background) as black.

        # 3. Apply the LUT using fast NumPy indexing.
        # This maps each label in `convergence_regions` to its corresponding color in `lut`.
        convergence_viz = lut[convergence_regions]
        
    # Blend the original image with the overlays
    cimg = cv2.addWeighted(cimg, 0.6, convergence_viz.astype(np.uint8), 0.4, 0)
    cimg[:,:,0] = cv2.add(cimg[:,:,0], acc_viz) # Add accumulator votes in blue
    
    plt.figure(figsize=(12, 8))
    plt.imshow(cimg)
    if pts.size > 0:
        plt.plot(pts[:,0], pts[:,1], 'r+', markersize=12, markeredgewidth=2, label='Detected Corners')
    plt.title("ASN Corner Detection Results")
    plt.legend()

def main():
    parser = argparse.ArgumentParser(description="Run ASN corner detector on an image.")
    parser.add_argument("image_path", nargs='?', default="./raw_camera_0_index_0.png", help="Path to the input image.")
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {args.image_path}")
        return

    # Example config. Try toggling use_fast_square_integration to see the performance difference.
    config = ASNConfig(threshold_min_nb_solutions=20, use_fast_square_integration=False)

    print("Running ASN detector...")
    start_detection = time.perf_counter()
    pts, accumulateur, convergence_regions = ASN_detector(img, config)
    end_detection = time.perf_counter()
    print(f"ASN detection took: {end_detection - start_detection:.4f} seconds")

    print("Preparing visualization...")
    start_display = time.perf_counter()
    display_results(img, pts, accumulateur, convergence_regions)
    end_display = time.perf_counter()
    print(f"Display preparation took: {end_display - start_display:.4f} seconds")

    print("Showing plot. Close the window to exit.")
    plt.show()

if __name__ == "__main__":
    main()
