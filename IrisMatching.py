import numpy as np
import random
from scipy.spatial import distance
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import LocallyLinearEmbedding

def selectRandomTestSample(test_features, test_labels):
    random_indices = random.sample(range(len(test_labels)), 108)
    sampled_features = test_features[random_indices, :]
    sampled_labels = test_labels[random_indices]
    return sampled_features, sampled_labels

def calculateMinimumDistance(training_feature, test_feature_sample, shift_offsets, distance_metric):
    return min(distance_metric(training_feature, np.roll(test_feature_sample, shift)) for shift in shift_offsets)

def evaluateTestSample(training_features, training_labels, single_test_feature, true_test_label, distance_type):
    shift_offsets = np.arange(-10, 11, 2)
    distance_metrics = [distance.cityblock, distance.euclidean, distance.cosine]

    all_distances = [calculateMinimumDistance(training_feature, single_test_feature, shift_offsets, distance_metrics[distance_type - 1])
                     for training_feature in training_features]

    minimum_distance = np.min(all_distances)
    predicted_label = training_labels[np.argmin(all_distances)]
    matching_distances = [dist for dist, label in zip(all_distances, training_labels) if label == true_test_label]
    non_matching_distances = [dist for dist, label in zip(all_distances, training_labels) if label != true_test_label]

    return predicted_label, matching_distances, non_matching_distances

def IrisMatching(training_set_features, training_set_labels, testing_set_features, testing_set_labels, distance_type):
    count_correct_matches = 0
    matched_distances_list, non_matched_distances_list = [], []

    for test_feature, test_label in zip(testing_set_features, testing_set_labels):
        predicted_label, matched_distances, non_matched_distances = evaluateTestSample(
            training_set_features, training_set_labels, test_feature, test_label, distance_type
        )
        
        matched_distances_list.extend(matched_distances)
        non_matched_distances_list.extend(non_matched_distances)
        if predicted_label == test_label:
            count_correct_matches += 1

    correct_match_rate = count_correct_matches / len(testing_set_labels)
    return correct_match_rate, matched_distances_list, non_matched_distances_list

def IrisMatchingWithDimensionalityReduction(training_features, training_labels, test_features, test_labels, components_count):
    dimensionality_reducer = (LinearDiscriminantAnalysis(n_components=components_count) if components_count < 108 else
               LocallyLinearEmbedding(n_neighbors=components_count + 1, n_components=components_count) if components_count < 323 else
               None)

    if dimensionality_reducer:
        training_features = dimensionality_reducer.fit_transform(training_features, training_labels)
        test_features = dimensionality_reducer.transform(test_features)

    match_rates = []
    for metric in ['l1', 'l2', 'cosine']:
        classifier = KNeighborsClassifier(n_neighbors=1, metric=metric)
        classifier.fit(training_features, training_labels)
        match_rate = np.mean(classifier.predict(test_features) == test_labels)
        match_rates.append(match_rate)

    return tuple(match_rates)

def computeROCCurve(matching_distances, non_matching_distances, threshold_values):
    false_match_rates = []
    false_non_match_rates = []

    for threshold in threshold_values:
        false_non_match_rate = np.mean(np.array(matching_distances) > threshold)
        false_match_rate = np.mean(np.array(non_matching_distances) < threshold)
        false_non_match_rates.append(false_non_match_rate)
        false_match_rates.append(false_match_rate)

    return false_match_rates, false_non_match_rates
