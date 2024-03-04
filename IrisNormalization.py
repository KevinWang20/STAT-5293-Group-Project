import math
import numpy as np

def IrisNormalization(image, pupil_circle, iris_circle):
    original_img = image
    radial_res = 64
    angular_res = 512
    iris_normalized = np.zeros((radial_res, angular_res), np.uint8)
    pupil_center_y = pupil_circle[0]  
    pupil_center_x = pupil_circle[1]  
    iris_center_y = iris_circle[0]
    iris_center_x = iris_circle[1]

    angle_step = 2.0 * math.pi / angular_res
    theta_values = np.arange(0, 2.0 * math.pi, angle_step)
    pupil_boundary_x = np.zeros((1, angular_res))
    pupil_boundary_y = np.zeros((1, angular_res))
    iris_boundary_x = np.zeros((1, angular_res))
    iris_boundary_y = np.zeros((1, angular_res))

    for j in range(angular_res):
        theta = theta_values[j]

        pupil_boundary_x[0, j] = pupil_center_y + pupil_circle[2] * math.cos(theta)
        pupil_boundary_y[0, j] = pupil_center_x + pupil_circle[2] * math.sin(theta)
        iris_boundary_x[0, j] = iris_center_y + iris_circle[2] * math.cos(theta)
        iris_boundary_y[0, j] = iris_center_x + iris_circle[2] * math.sin(theta)

    for i in range(radial_res):
        for j in range(angular_res):
            interp_pupil_iris_y = (iris_boundary_y[0, j] - pupil_boundary_y[0, j]) * (i / radial_res) + pupil_boundary_y[0, j]
            interp_pupil_iris_x = (iris_boundary_x[0, j] - pupil_boundary_x[0, j]) * (i / radial_res) + pupil_boundary_x[0, j]

            # Ensure interpolated coordinates are within image bounds
            valid_y = min(max(int(round(interp_pupil_iris_y)), 0), original_img.shape[0] - 1)
            valid_x = min(max(int(round(interp_pupil_iris_x)), 0), original_img.shape[1] - 1)

            iris_normalized[i, j] = original_img[valid_y, valid_x]

    inverted_image = 255 - iris_normalized
    return inverted_image
