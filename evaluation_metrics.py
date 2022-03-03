

# -*- coding: utf-8 -*-

"""
Master's Degree Final Project
Theme: Clinical Data Mining and Classification
Python version: 3.7.8

@author: Adara Nogueira
"""

#==============================================================================#
#==============================================================================#
# #                                MODULES                                   # #
#==============================================================================#
#==============================================================================#

import numpy as np

import load_data
import split_data

#==============================================================================#
#==============================================================================#
# #                               METHODS                                    # #
#==============================================================================#
#==============================================================================#

#.................................. MAIN METHOD ...............................#

def calculate(name, confusion_matrix):
    """
    Calculate evaluation metrics given a confusion matrix

    Arguments:
    confusion_matrix -- confusion matrix
    
    Returns:
    evaluation -- evaluation metrics calculated
    """

    # Count true positive, false positive, true negative, and false negative
    tn, fp, fn, tp = count_pos_neg(confusion_matrix)

    # All evaluation metrics
    evaluation = {"name":name,
                  "accuracy":accuracy(confusion_matrix),
                  "fnr":fnr(fn, tp),
                  "tnr":tnr(tn, fp),
                  "tpr":tpr(tp, fn),
                  "fpr":fpr(fp, tn),
                  "error":error(confusion_matrix),
                  "precision":precision(tp, fp),
                  "f_measure":f_measure(precision(tp, fp), tpr(tp, fn)),
                  "confusion_matrix":confusion_matrix,
                  "tn":tn,
                  "fp":fp,
                  "fn":fn,
                  "tp":tp}
    return evaluation

#................................ NEGATIVE RATES ..............................#

def tnr(tn, fp):
    """
    True negative rate (specificity)

    Arguments:
    fp -- number of false positives
    tn -- number of true negatives
    
    Returns:
    tnr
    """
    
    return tn / (tn + fp)

def fnr(fn, tp):
    """
    False negative rate (miss rate)

    Arguments:
    tp -- number of true positives
    fn -- number of false negatives
    
    Returns:
    fnr
    """
    
    return fn / (fn + tp) # 1 - tpr

#................................ POSITIVE RATES ..............................#

def tpr(tp, fn):
    """
    True positive rate (hit rate or recall)

    Arguments:
    tp -- number of true positives
    fn -- number of false negatives
    
    Returns:
    tpr
    """
    
    return tp / (tp + fn)

def fpr(fp, tn):
    """
    False positive rate (false alarm rate)

    Arguments:
    fp -- number of false positives
    tn -- number of true negatives
    
    Returns:
    fpr
    """
    
    return fp / (fp + tn)

#............................... CONFUSION MATRIX .............................#

def map_label_index(y):
    """
    Map class label to index
    Make all (0,0) position in the confusion matrix be non-cancerous (where applicable)
    (true negative)

    Arguments:
    y -- class label vector
    
    Returns:
    dictionary where the key is a class label and the value is an index
    """
    
    unique = np.unique(y)
    if unique[0] == 'Cancer':
        # Dataset: Ovarian
        y = unique[::-1]
    elif len(unique) == 5:
        # Dataset: Lung
        y = np.array([2.0, 1.0, 3.0, 4.0, 5.0])
    else:
        # Dataset: all others
        y = unique
    return dict(zip(y, range(len(y))))
    
def initiate_confusion_matrix(y):
    """
    Apply the feature selection technique to the data

    Arguments:
    y -- class label vector
    
    Returns:
    confusion matrix initialized with all values set to zero
    """

    number_of_classes = len(np.unique(y))
    return np.zeros((number_of_classes, number_of_classes)), map_label_index(y)

def count_pos_neg(confusion_matrix):
    """
    Calculate the number of true positive, false positive, true negative, and false negative

    Arguments:
    confusion_matrix -- confusion matrix
    
    Returns:
    tn -- number of negative instances classified as negative
    fp -- number of negative instances classified as positive (error)
    fn -- number of positive instances classified as negative (error)
    tp -- number of positive instances classified as positive
    """

    tn = np.sum(confusion_matrix[:1,0])     # (0,0)
    fp = np.sum(confusion_matrix[:1,1:])    # (0,1)
    fn = np.sum(confusion_matrix[1:,0])     # (1,0)
    tp = np.sum(confusion_matrix[1:,1:])    # (1,1)
    return tn, fp, fn, tp

#................................... OTHERS ...................................#

def accuracy(confusion_matrix):
    """
    Calculate the accuracy of a classifier

    Arguments:
    confusion_matrix -- confusion matrix
    
    Returns:
    accuracy
    """

    # Number of correct classifications
    diagonal = sum(confusion_matrix.diagonal())

    # Total number of classifications
    total = confusion_matrix.sum()

    # Compute accuracy
    return diagonal/total

def error(confusion_matrix):
    """
    Calculate the error rate

    Arguments:
    confusion_matrix -- confusion matrix
    
    Returns:
    error
    """

    # Compute error
    return 1 - accuracy(confusion_matrix)

def precision(tp, fp):
    """
    Calculate the precision (proportion of positive instances that actually are postive)

    Arguments:
    tp -- number of true positives
    fp -- number of false positives
    
    Returns:
    precision
    """

    # Compute precision
    return tp / (tp + fp)

def f_measure(precision, recall):
    """
    The F-measure is the harmonic mean of precision and recall

    Arguments:
    precision
    recall
    
    Returns:
    F-measure
    """

    # Compute F-measure
    return 2 / ((1/precision) + (1/recall))

if __name__ == "__main__":
    # Load data
    files_names = load_data.data_files_names
    datas, names = load_data.apply(files_names)
    
    # Go through all datasets
    for d, (data, data_name) in enumerate(zip(datas, names)):
        print(">> Dataset: " + str(data_name) + " <<\n")
        
        # Split the dataset into features and class
        _, y = split_data.features_class_split(data)

        mapping = map_label_index(y)

        print("y", np.unique(y))
        print("mapping", mapping, "\n")
