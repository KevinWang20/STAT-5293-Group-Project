import numpy as np
import cv2
from skimage.transform import hough_circle, hough_circle_peaks

def detect_pupil_center(eye_img, pupil_x, pupil_y):
    pupil_region = eye_img[pupil_y - 60:pupil_y + 60, pupil_x - 60:pupil_x + 60]
    smoothed_region = cv2.GaussianBlur(pupil_region, (5, 5), 0)
    circles = cv2.HoughCircles(
        smoothed_region, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2,
        minDist=200,
        param1=200,
        param2=12,
        minRadius=15,
        maxRadius=80
    )
    pupil_parameters = np.round(circles[0][0]).astype("int")
    return pupil_parameters

def IrisLocalization(eye_image):
    smoothed_eye = cv2.bilateralFilter(eye_image, 9, 75, 75)
    min_x = np.argmin(smoothed_eye.sum(axis=0))
    min_y = np.argmin(smoothed_eye.sum(axis=1))
    
    search_region = smoothed_eye[
        max(min_y - 60, 0):min(min_y + 60, eye_image.shape[0]),
        max(min_x - 60, 0):min(min_x + 60, eye_image.shape[1])
    ]
    
    offset_x = np.argmin(search_region.sum(axis=0))
    offset_y = np.argmin(search_region.sum(axis=1))
    pupil_x, pupil_y = min_x - 60 + offset_x, min_y - 60 + offset_y
    
    if pupil_x >= 100 and pupil_y >= 80:
        center_x, center_y, radius_pupil = detect_pupil_center(eye_image, pupil_x, pupil_y)
    else:
        circles = cv2.HoughCircles(
            smoothed_eye, 
            cv2.HOUGH_GRADIENT, 
            4, 
            280, 
            minRadius=25, 
            maxRadius=55, 
            param2=51
        )
        center_x, center_y, radius_pupil = np.round(circles[0][0]).astype("int")
    
    center_x += (pupil_x - 60)
    center_y += (pupil_y - 60)
    
    radius_pupil += 7
    eye_blurred = cv2.medianBlur(eye_image, 11)
    edges_eye = cv2.Canny(eye_blurred, threshold1=15, threshold2=30, L2gradient=True)
    edges_eye[:, center_x - radius_pupil - 30:center_x + radius_pupil + 30] = 0
    
    iris_radii = np.arange(radius_pupil + 45, 150, 2)
    hough_results = hough_circle(edges_eye, iris_radii)
    accumulators, circles_x, circles_y, radii = hough_circle_peaks(hough_results, iris_radii, total_num_peaks=1)
    iris_details = [circles_x[0], circles_y[0], radii[0]]
    
    if np.sqrt((iris_details[0] - center_x) ** 2 + (iris_details[1] - center_y) ** 2) > radius_pupil * 0.3:
        iris_details[0:2] = [center_x, center_y]
    
    return np.array(iris_details), np.array([center_x, center_y, radius_pupil])


