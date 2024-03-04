from scipy import ndimage
import math
import numpy as np


def defined_filter(x, y, f):
    return math.cos(2 * math.pi * f * math.sqrt(x**2 * y**2))


def create_space_constants():
    return [(3, 1.5), (4.5, 1.5)]


def create_frequencies():
    return [0.1, 0.07]


def create_ranges():
    return [range(-9, 10), range(-14, 15)], range(-5, 6)


def gabor_filter(x, y, space_constant_x, space_constant_y, f):
    exponential = np.exp(-0.5 * (x**2 / space_constant_x**2 + y**2 / space_constant_y**2))
    return (1 / (2 * math.pi * space_constant_x * space_constant_y)) * exponential * defined_filter(x, y, f)


def create_filters(x_ranges, y_range, space_constants, frequencies):
    filters = []
    for x_range, (space_constant_x, space_constant_y), f in zip(x_ranges, space_constants, frequencies):
        filter_matrix = [[gabor_filter(x, y, space_constant_x, space_constant_y, f) for x in x_range] for y in y_range]
        filters.append(np.real(np.reshape(filter_matrix, (len(y_range), len(x_range)))))
    return filters


def convolve_roi_with_filters(roi, filters):
    return [ndimage.convolve(roi, filter_matrix, mode='wrap', cval=0) for filter_matrix in filters]


def calculate_vector(filtered_eye, block_size):
    vector = []
    for i in range(0, filtered_eye.shape[0], block_size):
        for j in range(0, filtered_eye.shape[1], block_size):
            block = filtered_eye[i:i + block_size, j:j + block_size]
            mean_val = np.mean(block)
            aad = np.mean(np.abs(block - mean_val))
            vector.extend([mean_val, aad])
    return vector


def IrisFeatureExtraction(roi):
    x_ranges, y_range = create_ranges()
    space_constants = create_space_constants()
    frequencies = create_frequencies()
    
    filters = create_filters(x_ranges, y_range, space_constants, frequencies)
    filtered_images = convolve_roi_with_filters(roi, filters)
    
    vector = []
    for filtered_image in filtered_images:
        vector.extend(calculate_vector(filtered_image, block_size=8))
    
    return np.array(vector)



