import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from scipy.ndimage import maximum_filter

#copyright intellar@intellar.ca

@dataclass
class ASNConfig:
    """Configuration for the ASN detector."""
    threshold_min_nb_solutions: int = 20
    threshold_sol_determinant: float = 1e-6
    threshold_ratio_eigenvalues: float = 0.1
    sobel_kernel_size: int = 7
    integration_size: int = 25
    threshold_min_area_size: int = 0 # New: Threshold for the number of unique contributors
    gaussian_blur_sigma: float = 0.0 # New: Sigma for pre-smoothing Gaussian blur. 0 to disable.
    use_fast_square_integration: bool = False # If True, uses a fast box filter instead of a disk kernel.

def _disk_kernel(h: int, w: int) -> np.ndarray:
    """Creates a circular disk-shaped kernel."""
    if h <= 0 or w <= 0:
        raise ValueError("Height and width must be positive.")
    
    center_y, center_x = h // 2, w // 2
    radius = min(center_x, center_y, w - center_x, h - center_y)
    y, x = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
    mask = (x*x + y*y <= radius*radius).astype(float)
    return mask

def _sanitize_determinant(
    DetA: np.ndarray, 
    Dx2: np.ndarray, 
    Dy2: np.ndarray, 
    DxDy: np.ndarray, 
    threshold_sol_determinant: float, 
    threshold_ratio_eigenvalues: float
) -> np.ndarray:

    DetA[np.abs(DetA)<threshold_sol_determinant] = np.inf
    
    # --- Improved Eigenvalue Calculation ---
    # Calculate eigenvalues from the trace and determinant of the structure tensor.
    # This is more direct and numerically stable.
    trace = Dx2 + Dy2
    # The determinant is already calculated as DetA, but we use it here before it's sanitized.
    det = Dx2 * Dy2 - DxDy**2
    
    # Eigenvalues are (trace +/- sqrt(trace^2 - 4*det)) / 2
    sqrt_discriminant = np.sqrt(np.maximum(0, trace**2 - 4 * det)) # Use np.maximum to avoid sqrt of negative
    lambda1 = (trace + sqrt_discriminant) / 2
    lambda2 = (trace - sqrt_discriminant) / 2

    # Filter based on eigenvalue ratio to remove responses on straight edges.
    # Using np.divide with a small epsilon avoids division by zero.
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.divide(lambda1, lambda2, out=np.full_like(lambda1, np.inf), where=lambda2>1e-9)

    DetA[ratio < threshold_ratio_eigenvalues] = np.inf
    DetA[ratio > 1 / threshold_ratio_eigenvalues] = np.inf
    return DetA
    
def _calculate_structure_tensors(img: np.ndarray, config: ASNConfig) -> Tuple:
    """Calculates the structure tensor components for the image."""
    h, w = img.shape

    # --- Optional Pre-smoothing for Noise Robustness ---
    smoothed_img = img
    if config.gaussian_blur_sigma > 0:
        ksize = int(6 * config.gaussian_blur_sigma) // 2 * 2 + 1 # Ensure odd kernel size
        smoothed_img = cv2.GaussianBlur(img, (ksize, ksize), config.gaussian_blur_sigma)
    
    ddepth = cv2.CV_32F
    scale = 1
    delta = 0
    dx = cv2.Sobel(smoothed_img, ddepth, 1, 0, ksize=config.sobel_kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    dy = cv2.Sobel(smoothed_img, ddepth, 0, 1, ksize=config.sobel_kernel_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    dxdy = dx*dy
    dx2 = dx*dx 
    dy2 = dy*dy
    ddepth = -1 # Output array has the same depth as the input

    if not config.use_fast_square_integration:
        # --- High-Quality Disk Kernel Integration ---
        integration_windows = _disk_kernel(config.integration_size, config.integration_size)
        kernel_sum = np.sum(integration_windows)
        if kernel_sum > 0:
            integration_windows /= kernel_sum
        
        Dx2 = cv2.filter2D(dx*dx,ddepth,integration_windows)
        Dy2 = cv2.filter2D(dy*dy,ddepth,integration_windows)
        DxDy = cv2.filter2D(dxdy,ddepth,integration_windows)
        DxDyX = cv2.filter2D(dxdy*x,ddepth,integration_windows)
        DxDyY = cv2.filter2D(dxdy*y,ddepth,integration_windows)
        Dx2X = cv2.filter2D(dx2*x,ddepth,integration_windows)
        Dy2Y = cv2.filter2D(dy2*y,ddepth,integration_windows)
    else:
        # --- Fast Square Kernel Integration (Box Filter) ---
        ksize = (config.integration_size, config.integration_size)
        # The `normalize=True` argument makes it a mean filter, equivalent to the normalized disk.
        filter_func = lambda src: cv2.boxFilter(src, ddepth, ksize, normalize=True)
        Dx2, Dy2, DxDy = filter_func(dx2), filter_func(dy2), filter_func(dxdy)
        DxDyX, DxDyY = filter_func(dxdy*x), filter_func(dxdy*y)
        Dx2X, Dy2Y = filter_func(dx2*x), filter_func(dy2*y)
    
    return Dx2, Dy2, DxDy, Dx2X, Dy2Y, DxDyX, DxDyY

def _accumulate_votes(
    localSolutionX: np.ndarray, 
    localSolutionY: np.ndarray,
    img_shape: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Accumulates votes in an accumulator array using sub-pixel voting."""
    h, w = img_shape
    
    # Create a mask for valid solutions
    valid_mask = np.isfinite(localSolutionX) & np.isfinite(localSolutionY)
    valid_mask &= (localSolutionX > 1) & (localSolutionX < w - 1)
    valid_mask &= (localSolutionY > 1) & (localSolutionY < h - 1)

    # Get valid solutions and their original indices
    sol_x = localSolutionX[valid_mask]
    sol_y = localSolutionY[valid_mask]
    
    # Integer and fractional parts for sub-pixel vote distribution
    sol_x_i = sol_x.astype(int)
    sol_y_i = sol_y.astype(int)
    sol_x_f = sol_x - sol_x_i
    sol_y_f = sol_y - sol_y_i

    # Weights for the 4-cell neighborhood
    w1 = (1 - sol_x_f) * (1 - sol_y_f)
    w2 = (1 - sol_x_f) * sol_y_f
    w3 = sol_x_f * (1 - sol_y_f)
    w4 = sol_x_f * sol_y_f

    # Accumulate votes using np.add.at for atomic adds
    accumulateur = np.zeros(img_shape)
    np.add.at(accumulateur, (sol_y_i, sol_x_i), w1)
    np.add.at(accumulateur, (sol_y_i + 1, sol_x_i), w2)
    np.add.at(accumulateur, (sol_y_i, sol_x_i + 1), w3)
    np.add.at(accumulateur, (sol_y_i + 1, sol_x_i + 1), w4)

    original_indices = np.flatnonzero(valid_mask)
    # Map original pixel indices to the 4 accumulator cells they vote for
    acc_indices_tl = np.ravel_multi_index((sol_y_i, sol_x_i), img_shape)
    acc_indices_bl = np.ravel_multi_index((sol_y_i + 1, sol_x_i), img_shape)
    acc_indices_tr = np.ravel_multi_index((sol_y_i, sol_x_i + 1), img_shape)
    acc_indices_br = np.ravel_multi_index((sol_y_i + 1, sol_x_i + 1), img_shape)
    
    # Concatenate all mappings
    all_acc_indices = np.concatenate([acc_indices_tl, acc_indices_bl, acc_indices_tr, acc_indices_br])
    all_original_indices = np.concatenate([original_indices] * 4)

    return accumulateur, all_acc_indices, all_original_indices

def _find_and_refine_peaks(
    accumulateur: np.ndarray,
    acc_indices: np.ndarray,
    original_indices: np.ndarray,
    tensors: Tuple,
    config: ASNConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """Finds local maxima in the accumulator and refines their positions."""
    img_shape = accumulateur.shape
    Dx2, Dy2, DxDy, Dx2X, Dy2Y, DxDyX, DxDyY = tensors
    
    # --- Vectorized Local Maxima Detection ---
    local_max = maximum_filter(accumulateur, size=3, mode='constant')
    # A point is a local max if it's equal to the max in its 3x3 neighborhood
    # and it's greater than the minimum threshold
    is_peak = (accumulateur == local_max) & (accumulateur > config.threshold_min_nb_solutions)
    
    # Ensure peaks are not on the border
    is_peak[:1, :] = is_peak[-1:, :] = is_peak[:, :1] = is_peak[:, -1:] = False
    
    peak_coords = np.argwhere(is_peak)
    
    pts = []
    convergence_regions = np.zeros(img_shape)

    for i, j in peak_coords:
        # --- ASN Refinement ---
        # Get indices of the 3x3 neighborhood in the accumulator
        y_range, x_range = np.mgrid[i-1:i+2, j-1:j+2]
        neighborhood_indices = np.ravel_multi_index((y_range, x_range), img_shape).flatten()
        
        # Find all original pixels that contributed to this 3x3 region
        mask_contrib = np.isin(acc_indices, neighborhood_indices)
        contrib_indices_flat = np.unique(original_indices[mask_contrib])

        # Validate against the number of unique contributors (the "area of attraction")
        if contrib_indices_flat.size < config.threshold_min_area_size:
            continue
        
        if contrib_indices_flat.size == 0:
            continue
        
        contrib_y, contrib_x = np.unravel_index(contrib_indices_flat, img_shape)
        
        # Sum up the matrix values from all contributors
        dx2_ = Dx2[contrib_y, contrib_x].sum()
        dx2x_ = Dx2X[contrib_y, contrib_x].sum()
        dy2_ = Dy2[contrib_y, contrib_x].sum()
        dy2y_ = Dy2Y[contrib_y, contrib_x].sum()
        dxdy_ = DxDy[contrib_y, contrib_x].sum()
        dxdyx_ = DxDyX[contrib_y, contrib_x].sum()
        dxdyy_ = DxDyY[contrib_y, contrib_x].sum()
        
        convergence_regions[contrib_y, contrib_x] = 255
        
        detA_ = dx2_ * dy2_ - dxdy_**2
        if abs(detA_) > 1e-9:
            solution_X = (dy2_*(dx2x_+dxdyy_)-dxdy_*(dy2y_+dxdyx_))/(detA_)
            solution_Y = (dx2_*(dy2y_+dxdyx_)-dxdy_*(dx2x_+dxdyy_))/(detA_)
            pts.append([solution_X, solution_Y])

    return np.array(pts), convergence_regions

def ASN_detector(img: np.ndarray, config: ASNConfig = ASNConfig()) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detects corners in an image using the ASN (Aggregated Solution Normalization) operator.

    Args:
        img: Grayscale input image as a NumPy array.
        config: Configuration object for the detector parameters.

    Returns:
        A tuple containing:
        - pts: An array of [x, y] coordinates for detected corners.
        - accumulateur: The accumulator array showing vote distribution.
        - convergence_regions: An image highlighting pixels that contributed to corners.
    """
    tensors = _calculate_structure_tensors(img, config)
    Dx2, Dy2, DxDy, Dx2X, Dy2Y, DxDyX, DxDyY = tensors
    
    DetA = Dx2*Dy2 - DxDy**2
    DetA = _sanitize_determinant(
        DetA, Dx2, Dy2, DxDy, 
        config.threshold_sol_determinant, 
        config.threshold_ratio_eigenvalues
    )
    
    localSolutionX = (Dy2*(Dx2X+DxDyY)-DxDy*(Dy2Y+DxDyX))/(DetA)
    localSolutionY = (Dx2*(Dy2Y+DxDyX)-DxDy*(Dx2X+DxDyY))/(DetA)
    
    accumulateur, acc_indices, original_indices = _accumulate_votes(localSolutionX, localSolutionY, img.shape)
    
    pts, convergence_regions = _find_and_refine_peaks(accumulateur, acc_indices, original_indices, tensors, config)
    
    return pts, accumulateur, convergence_regions
