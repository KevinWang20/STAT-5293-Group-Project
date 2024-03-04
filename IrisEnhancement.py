from skimage.filters.rank import equalize
from skimage.morphology import disk
import numpy as np

def IrisEnhancement(NormalizedIris):
    # Ensure the input is a numpy array
    if not isinstance(NormalizedIris, np.ndarray):
        raise ValueError("The input NormalizedIris must be a numpy array.")
    
    # Ensure the array is two-dimensional
    if NormalizedIris.ndim != 2:
        raise ValueError("The input NormalizedIris must be a two-dimensional array.")
    
    # Convert the input image to the correct type if necessary
    if NormalizedIris.dtype != np.uint8:
        NormalizedIris_uint8 = NormalizedIris.astype(np.uint8)
    else:
        NormalizedIris_uint8 = NormalizedIris

    # Define the structural element for histogram equalization
    selem = disk(32)  # Create a disk-shaped structural element with radius 32

    # Perform histogram equalization on the image
    equalized_image = equalize(NormalizedIris_uint8, selem)

    # Define the region of interest (ROI)
    # We will only take the top 48 rows of the image
    roi_top_row = 0
    roi_bottom_row = 48  # This is exclusive
    roi_left_column = 0
    roi_right_column = equalized_image.shape[1]  # This is the width of the image

    # Extract the region of interest from the equalized image
    roi = equalized_image[roi_top_row:roi_bottom_row, roi_left_column:roi_right_column]

    # Return the region of interest
    return roi



