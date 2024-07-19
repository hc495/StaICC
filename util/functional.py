import math
import warnings
from . import configs

def exp_to_list(list):
    return [math.exp(x) for x in list]

def softmax(x):
    f_x_max = max(x)
    f_x = [x_i - f_x_max for x_i in x]
    f_x = exp_to_list(f_x) 
    sum_x = sum(f_x)
    return [x_i / sum_x for x_i in f_x]

def argmax(x):
    return max(range(len(x)), key=lambda i: x[i])

def unique_check(list):
    if len(list) != len(set(list)):
        return False
    else:
        return True
    
def linspace(start, end, num):
    if num <= 1:
        raise ValueError("num should be greater than 1.")
    return [start + (end - start) * i / (num - 1) for i in range(num)]
    
def extend_onehot_prediction_to_logits(prediction: list[int]) -> list[list[float]]:
    if type(prediction[0]) == list:
        return prediction
    if not all([0 <= x < len(prediction) for x in prediction]):
        raise ValueError("The prediction should be in the range of [0, len(prediction)).")
    
    return [[1 if i == x else 0 for i in range(len(prediction))] for x in prediction]
    
def accuracy(ground_truth: list[int], prediction):
    if len(ground_truth) != len(prediction):
        raise ValueError("The length of ground_truth and prediction should be the same.")
    if not all([all([0 <= y <= 1 for y in x]) for x in prediction]):
        raise ValueError("The prediction should be in the range of [0, 1].")
    
    correct = 0
    for i in range(len(ground_truth)):
        if argmax(prediction[i]) == ground_truth[i]:
            correct += 1
    return correct / len(ground_truth)

def averaged_truelabel_likelihood(ground_truth: list[int], prediction):
    if len(ground_truth) != len(prediction):
        raise ValueError("The length of ground_truth and prediction should be the same.")
    if not all([all([0 <= y <= 1 for y in x]) for x in prediction]):
        raise ValueError("The prediction should be in the range of [0, 1].")
    
    likelihood = 0
    for i in range(len(ground_truth)):
        likelihood += prediction[i][ground_truth[i]]
    return likelihood / len(ground_truth)

def macro_F1(ground_truth: list[int], prediction):
    if len(ground_truth) != len(prediction):
        raise ValueError("The length of ground_truth and prediction should be the same.")
    if not all([all([0 <= y <= 1 for y in x]) for x in prediction]):
        raise ValueError("The prediction should be in the range of [0, 1].")
    
    TP = [0] * len(prediction[0])
    FP = [0] * len(prediction[0])
    FN = [0] * len(prediction[0])
    for i in range(len(ground_truth)):
        if argmax(prediction[i]) == ground_truth[i]:
            TP[ground_truth[i]] += 1
        else:
            FP[argmax(prediction[i])] += 1
            FN[ground_truth[i]] += 1
    
    precision = [TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) != 0 else 0 for i in range(len(TP))]
    recall = [TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) != 0 else 0 for i in range(len(TP))]
    F1 = [2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0 for i in range(len(TP))]
    return sum(F1) / len(F1)

def expected_calibration_error_1(ground_truth: list[int], prediction, bins = configs.STANDARD_SETTINGS["ece_bins"]):
    if len(ground_truth) != len(prediction):
        raise ValueError("The length of ground_truth and prediction should be the same.")
    if not all([all([0 <= y <= 1 for y in x]) for x in prediction]):
        raise ValueError("The prediction should be in the range of [0, 1].")
    if bins <= 1:
        raise ValueError("bins should be greater than 1.")
    if bins != configs.STANDARD_SETTINGS["ece_bins"]:
        warnings.warn(configs.WARNING_SETTINGS["tampering"])
        if configs.STRICT_MODE:
            warnings.warn(configs.WARNING_SETTINGS["strict_mode_protect"])
            bins = configs.STANDARD_SETTINGS["ece_bins"]
    if bins > len(ground_truth):
        raise ValueError("bins should be less than the length of ground_truth.")
    
    bin_boundaries = linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = [max(x) for x in prediction]
    predicted_label = [argmax(x) for x in prediction]

    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = [confidence for confidence, label in zip(confidences, predicted_label) if bin_lower <= confidence < bin_upper]
        if len(in_bin) == 0:
            continue
        accuracy_in_bin = [1 if label == ground_truth[i] else 0 for i, label in enumerate(predicted_label) if bin_lower <= confidences[i] < bin_upper]
        ece += len(in_bin) / len(ground_truth) * abs(sum(accuracy_in_bin) / len(accuracy_in_bin) - sum(in_bin) / len(in_bin))
    return ece