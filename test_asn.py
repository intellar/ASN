import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
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

    # Create colored overlays
    cimg[:,:,0] = cv2.add(cimg[:,:,0], acc_viz) # Blue channel for accumulator
    cimg[:,:,1] = cv2.add(cimg[:,:,1], (convergence_regions * 0.5).astype(np.uint8)) # Green for convergence

    plt.figure(figsize=(12, 8))
    plt.imshow(cimg)
    if pts.size > 0:
        plt.plot(pts[:,0], pts[:,1], 'r+', markersize=12, markeredgewidth=2, label='Detected Corners')
    plt.title("ASN Corner Detection Results")
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Run ASN corner detector on an image.")
    parser.add_argument("image_path", nargs='?', default="./raw_camera_0_index_0.png", help="Path to the input image.")
    args = parser.parse_args()

    img = cv2.imread(args.image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: Could not load image at {args.image_path}")
        return

    config = ASNConfig(threshold_min_nb_solutions=20)
    pts, accumulateur, convergence_regions = ASN_detector(img, config)
    display_results(img, pts, accumulateur, convergence_regions)

if __name__ == "__main__":
    main()
