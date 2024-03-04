from IrisMatching import IrisMatching, IrisMatchingWithDimensionalityReduction, computeROCCurve
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np

def calculateRecognitionRates(training_features, training_labels, testing_features, testing_labels):
    threshold_values = np.arange(0.04, 0.1, 0.003)
    recognition_rates = {}
    for metric_id in [1, 2, 3]:
        recognition_rates[metric_id] = IrisMatching(training_features, training_labels, testing_features, testing_labels, metric_id)[:3]
    recognition_rates[4], recognition_rates[5], recognition_rates[6] = IrisMatchingWithDimensionalityReduction(training_features, training_labels, testing_features, testing_labels, 200)
    return recognition_rates

def plotROCcurve(distances_match, distances_nonmatch, threshold_values):
    false_match_rates, false_non_match_rates = computeROCCurve(distances_match, distances_nonmatch, threshold_values)
    plt.plot(false_match_rates, false_non_match_rates)
    plt.xlabel('False Match Rate')
    plt.ylabel('False Non-match Rate')
    plt.title('ROC Curve')
    plt.savefig('roc_curve.png')
    plt.show()

def generateCRRtable(recognition_rates):
    print("Correct recognition rate (%)")
    comparison_table = [
        ['L1 distance measure', recognition_rates[1][0] * 100, recognition_rates[4] * 100],
        ['L2 distance measure', recognition_rates[2][0] * 100, recognition_rates[5] * 100],
        ['Cosine similarity measure', recognition_rates[3][0] * 100, recognition_rates[6] * 100]
    ]
    print(tabulate(comparison_table, headers=['Similarity measure', 'Original feature set', 'Reduced feature set']))

def plotPerformanceEvaluation(training_features, training_labels, testing_features, testing_labels):
    dimension_range = list(range(20,109,10))
    cosine_recognition_rates = []
    for feature_dimension in dimension_range:
        _, _, cosine_rate = IrisMatchingWithDimensionalityReduction(training_features, training_labels, testing_features, testing_labels, feature_dimension)
        cosine_recognition_rates.append(cosine_rate * 100)
    plt.plot(dimension_range, cosine_recognition_rates, marker="*", color='navy')
    plt.xlabel('Dimensionality of the feature vector')
    plt.ylabel('Correct Recognition Rate')
    plt.savefig('performance_evaluation.png')
    plt.show()

def generateFMandFNMtable(mean_false_match_rates, lower_false_match_rates, upper_false_match_rates, mean_false_non_match_rates, lower_false_non_match_rates, upper_false_non_match_rates, threshold_values):
    print("False Match and False Nonmatch Rates with Different Threshold Values")
    rates_table_data = [
        [
            threshold_values[7], 
            f"{mean_false_match_rates[7]}[{lower_false_match_rates[7]},{upper_false_match_rates[7]}]", 
            f"{mean_false_non_match_rates[7]}[{lower_false_non_match_rates[7]},{upper_false_non_match_rates[7]}]"
        ],
        # Repeat for other indices as needed...
    ]
    print(tabulate(rates_table_data, headers=['Threshold', 'False match rate(%)', "False non-match rate(%)"]))

# Main evaluation function with variable names updated
def tableCorrectRecognitionRates(training_features, training_labels, testing_features, testing_labels):
    recognition_rates = calculateRecognitionRates(training_features, training_labels, testing_features, testing_labels)
    generateCRRtable(recognition_rates)
    plotROCcurve(recognition_rates[3][1], recognition_rates[3][2], np.arange(0.04, 0.1, 0.003))

# Performance evaluation plotting function with variable names updated
def IrisPerformanceEvaluation(training_features, training_labels, testing_features, testing_labels):
    plotPerformanceEvaluation(training_features, training_labels, testing_features, testing_labels)

# Function for generating tables with updated variable names
def tableFalseMatchRates(mean_fm_rates, lower_fm_rates, upper_fm_rates, mean_fnm_rates, lower_fnm_rates, upper_fnm_rates, threshold_values):
    generateFMandFNMtable(mean_fm_rates, lower_fm_rates, upper_fm_rates, mean_fnm_rates, lower_fnm_rates, upper_fnm_rates, threshold_values)
