import os
import numpy as np
import matplotlib.pyplot as plt

from skimage import io, filters, morphology, transform, exposure

from skimage.filters import median
from skimage.morphology import disk


##############################################################################
# 1) Helper / Utility functions
##############################################################################

def ensure_out_folder_exists():
    """Ensure that an 'out' folder exists, create if it does not."""
    if not os.path.exists('out'):
        os.makedirs('out')


def float_to_uint8(img):
    """
    Convert a floating-point image (assumed in range [0, 1]) to uint8 [0..255].
    Values outside [0,1] are clipped.
    """
    # Ensure no negative or >1 values
    img = np.clip(img, 0, 1)
    # Scale and convert
    return (img * 255).astype(np.uint8)


def save_image(img, filename):
    """
    Saves an image (NumPy array) to the out/ folder with the specified filename.
    If the image is boolean, it will be cast to uint8 for saving.
    If the image is float, it will be cast to uint8 via float_to_uint8().
    """
    ensure_out_folder_exists()

    # Convert boolean masks to uint8 for saving
    if img.dtype == bool:
        img = img.astype(np.uint8) * 255

    # Convert floating-point images (like color_img) to uint8
    elif np.issubdtype(img.dtype, np.floating):
        img = float_to_uint8(img)

    io.imsave(os.path.join('out', filename), img)



##############################################################################
# 2) Image processing functions
##############################################################################

def load_image(path):
    """
    Loads a grayscale image from 'path' (e.g. 'brain.png').
    Returns a NumPy float64 array in the range [0, 1].
    """
    img = io.imread(path, as_gray=True)
    return img


def denoise_median(img, filter_size=3):
    """
    Applies a median filter to reduce salt & pepper noise.
    filter_size=3 uses a 3x3 structuring element (disk).
    """
    denoised = median(img, footprint=disk(filter_size))
    return denoised


def compute_otsu_threshold(img):
    """
    Computes Otsu's threshold for the given grayscale image.
    Returns the threshold (float).
    """
    return filters.threshold_otsu(img)


def create_masks(img, thresh_bg_brain, thresh_gm_wm):
    """
    Using two thresholds:
      - thresh_bg_brain: separate background vs. brain
      - thresh_gm_wm:    separate grey vs. white matter within the brain

    Returns three boolean masks:
      mask_bg: background
      mask_gm: grey matter
      mask_wm: white matter
    """
    # Mask 1: background vs. brain
    mask_bg = img < thresh_bg_brain
    mask_brain = ~mask_bg  # i.e., img >= thresh_bg_brain

    # Mask 2: within the brain region, separate grey vs. white
    # Only compute the second threshold within the "brain" region.
    # We already have that threshold (thresh_gm_wm), so:
    mask_gm = (img >= thresh_gm_wm) & mask_brain  # white matter or higher
    # but we want GM as an entire region below the threshold:
    # Actually, we have to be consistent with "grey matter < thresh" or > thresh
    # Typically, white matter is brighter, so:
    mask_wm = (img >= thresh_gm_wm) & mask_brain
    mask_gm = (img < thresh_gm_wm) & mask_brain

    return mask_bg, mask_gm, mask_wm


def plot_log_histogram(img, bins=256, title="Log-scaled Histogram"):
    """
    Plot a log-scaled histogram of image intensities.
    The function also returns the figure & axis so you can save or show it.
    """
    hist, bin_edges = np.histogram(img.flatten(), bins=bins, range=[0, 1])

    fig, ax = plt.subplots()
    ax.plot(bin_edges[:-1], hist, label='Histogram')
    ax.set_yscale('log')  # log scale on the y-axis
    ax.set_xlabel('Intensity')
    ax.set_ylabel('Count (log scale)')
    ax.set_title(title)
    ax.legend()

    return fig, ax


def combine_masks_to_color(mask_bg, mask_gm, mask_wm):
    """
    Combine three boolean masks (background, grey matter, white matter)
    into an RGB image:
      - background = Red
      - grey matter = Green
      - white matter = Blue
    """
    # Create an empty RGB image
    color_image = np.zeros((mask_bg.shape[0], mask_bg.shape[1], 3), dtype=np.float32)

    # Fill each channel with the appropriate mask
    color_image[mask_bg, 0] = 1.0  # R for background
    color_image[mask_gm, 1] = 1.0  # G for grey matter
    color_image[mask_wm, 2] = 1.0  # B for white matter

    return color_image


def create_border_between_gm_wm(mask_gm, mask_wm, selem_size=3):
    """
    Use a morphological operation to create a border between GM and WM.
    One simple approach is to erode or dilate one of the masks and
    look at the difference/overlap.

    Return a boolean mask that indicates the border pixels.
    """
    # Example: border as the boundary of the WM region
    # We can erode WM and then take the XOR with the original WM
    selem = morphology.disk(selem_size)
    wm_eroded = morphology.erosion(mask_wm, selem)
    border = mask_wm ^ wm_eroded

    # Alternatively, to find the exact boundary between GM and WM, we could do:
    # border = morphology.dilation(mask_gm, selem) & mask_wm
    # or many other approaches. Feel free to modify as needed.

    return border


def overlay_border_on_image(img, border_mask, color='red'):
    """
    Overlay a given border (boolean mask) on top of a grayscale image.
    color can be e.g. 'red', 'green', 'blue' to specify which channel to highlight.
    Returns an RGB image.
    """
    # Convert grayscale -> RGB
    rgb = np.dstack([img, img, img])

    # Decide which channel to highlight
    color_map = {
        'red': (0, 1, 2),  # highlight channel 0
        'green': (1, 0, 2),
        'blue': (2, 0, 1)
    }
    highlight_ch = color_map[color][0]

    # Make a copy so we don't modify original
    overlay = rgb.copy()
    # Increase the highlight channel for border pixels
    overlay[border_mask, highlight_ch] = 1.0

    return overlay


def upsample_image(img, scale_factor=4, method='bilinear'):
    """
    Upsample image by a factor along each axis.
    method can be 'bilinear' or 'nearest'.
    Returns the upsampled image.
    """
    # scikit-image's transform.resize expects output_shape
    out_rows = img.shape[0] * scale_factor
    out_cols = img.shape[1] * scale_factor

    # mode can be 'reflect', etc.  order=1 => bilinear, order=0 => nearest
    if method == 'bilinear':
        order = 1
    elif method == 'nearest':
        order = 0
    else:
        raise ValueError("Invalid interpolation method. Use 'bilinear' or 'nearest'.")

    resized = transform.resize(img, (out_rows, out_cols), order=order, preserve_range=True)
    return resized


##############################################################################
# 3) Main orchestration
##############################################################################

def main():
    # ------------------------------------------------------------------------
    # 0. Setup
    # ------------------------------------------------------------------------
    ensure_out_folder_exists()

    # ------------------------------------------------------------------------
    # 1. Load the noisy image
    # ------------------------------------------------------------------------
    img_path = 'brain-noisy.png'  # or 'brain-noisy.png' if that's your file
    img = load_image(img_path)

    # ------------------------------------------------------------------------
    # 2. Denoise via median filter
    # ------------------------------------------------------------------------
    denoised_img = denoise_median(img, filter_size=3)
    save_image(denoised_img, 'brain_denoised.png')

    # ------------------------------------------------------------------------
    # 3. Otsu Thresholding
    #    - First threshold to separate background from brain
    #    - Second threshold to separate grey matter from white matter
    # ------------------------------------------------------------------------
    # 3a. Global threshold for background vs brain
    thresh_bg_brain = compute_otsu_threshold(denoised_img)

    # 3b. Mask out brain region, compute threshold only within it
    #     Or simply compute once more but on the "brain" portion, etc.
    #     For demonstration, we'll just compute global Otsu again on brain region:
    brain_region = denoised_img[denoised_img >= thresh_bg_brain]
    thresh_gm_wm = filters.threshold_otsu(brain_region)

    # ------------------------------------------------------------------------
    # 4. Create three binary masks: background, grey matter, white matter
    # ------------------------------------------------------------------------
    mask_bg, mask_gm, mask_wm = create_masks(denoised_img,
                                             thresh_bg_brain,
                                             thresh_gm_wm)
    # Save these masks
    save_image(mask_bg, 'brain-bg.png')
    save_image(mask_gm, 'brain-gm.png')
    save_image(mask_wm, 'brain-wm.png')

    # ------------------------------------------------------------------------
    # 5. Plot a log-scaled histogram
    #    "How could you roughly estimate the two thresholds by looking at it?"
    # ------------------------------------------------------------------------
    fig, ax = plot_log_histogram(denoised_img, bins=256,
                                 title="Log-scaled Histogram of Denoised Image")
    fig.savefig(os.path.join('out', 'brain_histogram.png'))
    plt.close(fig)

    # (Answer to the question: by looking for the valleys in the histogram,
    #  you can often guess where to split the intensities.)

    # ------------------------------------------------------------------------
    # 6. Combine the three masks into a single colour image
    #    background=Red, grey=Green, white=Blue
    # ------------------------------------------------------------------------
    color_img = combine_masks_to_color(mask_bg, mask_gm, mask_wm)
    save_image(color_img, 'brain_color_map.png')  # This will save as an RGB image.

    # ------------------------------------------------------------------------
    # 7. Create a border between grey & white matter and overlay on denoised image
    # ------------------------------------------------------------------------
    border_mask = create_border_between_gm_wm(mask_gm, mask_wm, selem_size=3)
    # Overlay that border on the denoised image
    overlay_result = overlay_border_on_image(denoised_img, border_mask, color='red')
    save_image(overlay_result, 'brain_overlay_border.png')

    # ------------------------------------------------------------------------
    # 8. Upsample the image by factor of 4
    #    a) Bilinear interpolation
    #    b) Nearest neighbor interpolation
    #    Then compare segmentations
    # ------------------------------------------------------------------------
    # 8a. Bilinear
    up_bilinear_img = upsample_image(denoised_img, scale_factor=4, method='bilinear')
    save_image(up_bilinear_img, 'brain_upsampled_bilinear.png')

    # Compute new segmentation on the upsampled image
    # Re-run Otsu for background vs. brain, and for GM vs. WM
    thresh_bg_brain_bi = compute_otsu_threshold(up_bilinear_img)
    up_mask_bg_bi = up_bilinear_img < thresh_bg_brain_bi
    up_mask_brain_bi = ~up_mask_bg_bi

    # second threshold for GM vs. WM within brain region:
    up_brain_region_bi = up_bilinear_img[up_mask_brain_bi]
    thresh_gm_wm_bi = filters.threshold_otsu(up_brain_region_bi)
    up_mask_wm_bi = (up_bilinear_img >= thresh_gm_wm_bi) & up_mask_brain_bi
    up_mask_gm_bi = (up_bilinear_img < thresh_gm_wm_bi) & up_mask_brain_bi

    # 8b. Upsample the original masks by the same factor (compare)
    up_mask_bg_original_bi = upsample_image(mask_bg.astype(float), 4, 'bilinear') > 0.5
    up_mask_gm_original_bi = upsample_image(mask_gm.astype(float), 4, 'bilinear') > 0.5
    up_mask_wm_original_bi = upsample_image(mask_wm.astype(float), 4, 'bilinear') > 0.5

    # Save them
    save_image(up_mask_bg_original_bi, 'brain-bg_upsampled_bilinear.png')
    save_image(up_mask_gm_original_bi, 'brain-gm_upsampled_bilinear.png')
    save_image(up_mask_wm_original_bi, 'brain-wm_upsampled_bilinear.png')

    # Compare with masks from segmenting the already-upsampled image
    save_image(up_mask_bg_bi, 'brain-bg_segmented_on_upsampled_bilinear.png')
    save_image(up_mask_gm_bi, 'brain-gm_segmented_on_upsampled_bilinear.png')
    save_image(up_mask_wm_bi, 'brain-wm_segmented_on_upsampled_bilinear.png')

    # 8c. Repeat with Nearest Neighbor
    up_nearest_img = upsample_image(denoised_img, scale_factor=4, method='nearest')
    save_image(up_nearest_img, 'brain_upsampled_nearest.png')

    # Recompute thresholds on upsampled image
    thresh_bg_brain_nn = compute_otsu_threshold(up_nearest_img)
    up_mask_bg_nn = up_nearest_img < thresh_bg_brain_nn
    up_mask_brain_nn = ~up_mask_bg_nn
    up_brain_region_nn = up_nearest_img[up_mask_brain_nn]
    thresh_gm_wm_nn = filters.threshold_otsu(up_brain_region_nn)
    up_mask_wm_nn = (up_nearest_img >= thresh_gm_wm_nn) & up_mask_brain_nn
    up_mask_gm_nn = (up_nearest_img < thresh_gm_wm_nn) & up_mask_brain_nn

    # Upsample the original masks using nearest neighbor
    up_mask_bg_original_nn = upsample_image(mask_bg.astype(float), 4, 'nearest') > 0.5
    up_mask_gm_original_nn = upsample_image(mask_gm.astype(float), 4, 'nearest') > 0.5
    up_mask_wm_original_nn = upsample_image(mask_wm.astype(float), 4, 'nearest') > 0.5

    # Save them
    save_image(up_mask_bg_original_nn, 'brain-bg_upsampled_nearest.png')
    save_image(up_mask_gm_original_nn, 'brain-gm_upsampled_nearest.png')
    save_image(up_mask_wm_original_nn, 'brain-wm_upsampled_nearest.png')

    # Compare with nearest-neighbor segmented results
    save_image(up_mask_bg_nn, 'brain-bg_segmented_on_upsampled_nearest.png')
    save_image(up_mask_gm_nn, 'brain-gm_segmented_on_upsampled_nearest.png')
    save_image(up_mask_wm_nn, 'brain-wm_segmented_on_upsampled_nearest.png')

    # (Answer to the question about differences):
    # - When you upsample the original masks vs. thresholding again
    #   after upsampling the grayscale, you typically get differences
    #   because interpolation changes the intensities (especially bilinear),
    #   potentially shifting thresholds. Nearest neighbor preserves the
    #   exact 0/1 but changes shape boundaries.  Hence differences are more
    #   pronounced with bilinear interpolation.

    print("Processing completed. Check the 'out' folder for results.")


if __name__ == "__main__":
    main()
