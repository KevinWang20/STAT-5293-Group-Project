import numpy as np
import cv2
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from IrisEnhancement import IrisEnhancement
from IrisFeatureExtraction import IrisFeatureExtraction
import IrisMatching 
import IrisPerformanceEvaluation 
import datetime

def process_iris_images(root_path, num_subjects=108, num_train=3, num_test=4):
    train_features = np.zeros((num_subjects * num_train, 1536))
    train_classes = np.zeros(num_subjects * num_train, dtype=np.uint8)
    test_features = np.zeros((num_subjects * num_test, 1536))
    test_classes = np.zeros(num_subjects * num_test, dtype=np.uint8) 
    
    start_time = datetime.datetime.now()

    for subject_id in range(1, num_subjects + 1):
        base_path = f"{root_path}{str(subject_id).zfill(3)}"
        for session_id in range(1, num_train + 1):
            iris_path = f"{base_path}/1/{str(subject_id).zfill(3)}_1_{session_id}.bmp"
            img = cv2.imread(iris_path, 0)
            iris, pupil = IrisLocalization(img)
            normalized = IrisNormalization(img, pupil, iris)
            ROI = IrisEnhancement(normalized)
            index = (subject_id - 1) * num_train + session_id - 1
            train_features[index, :] = IrisFeatureExtraction(ROI)
            train_classes[index] = subject_id

        for session_id in range(1, num_test + 1):
            iris_path = f"{base_path}/2/{str(subject_id).zfill(3)}_2_{session_id}.bmp"
            img = cv2.imread(iris_path, 0)
            iris, pupil = IrisLocalization(img)
            normalized = IrisNormalization(img, pupil, iris)
            ROI = IrisEnhancement(normalized)
            index = (subject_id - 1) * num_test + session_id - 1
            test_features[index, :] = IrisFeatureExtraction(ROI)
            test_classes[index] = subject_id

    end_time = datetime.datetime.now()
    print(f'image processing and feature extraction takes {(end_time - start_time).seconds} seconds')

    return train_features, train_classes, test_features, test_classes

root_path = "./CASIA Iris Image Database (version 1.0)/"
train_features, train_classes, test_features, test_classes = process_iris_images(root_path)

IrisPerformanceEvaluation.tableCorrectRecognitionRates(train_features, train_classes, test_features, test_classes)
IrisPerformanceEvaluation.IrisPerformanceEvaluation(train_features, train_classes, test_features, test_classes)




