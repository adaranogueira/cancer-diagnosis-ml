

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

from skfeature.function.similarity_based import fisher_score
from skfeature.function.similarity_based import SPEC
from skfeature.function.similarity_based import lap_score
from skfeature.utility import construct_W
import numpy as np
import pickle

import data_classification
import data_representation
import evaluation_metrics
import pre_processing
import load_data
import split_data

#==============================================================================#
#==============================================================================#
# #                               METHODS                                    # #
#==============================================================================#
#==============================================================================#

#.................................. MAIN METHOD ...............................#

def apply(fs_method, X, y, *fs_args):
    """
    Apply the feature selection technique to the data
    Algorithm 5 of the master's thesis (Subsection 4.2.2)

    Arguments:
    fs_method -- feature selection technique
    X -- training set
    y -- actual class labels
    fs_args -- feature selection technique parameters
    
    Returns:
    indexes of selected features
    """

    print("- Dimensionality Reduction Technique: " + str(fs_method.__name__) + " <<\n")

    # Selected features
    return fs_method(X, y, *fs_args)

#............................ UNSUPERVISED METHODS ............................#

def laplacian_score(X, y=None):
    """
    Laplacian Score: similarity based feature selection

    Arguments:
    X -- training set
    y -- actual class labels (only for compatibility with the main code)
    
    Returns:
    indexes of selected features
    """
    
    # Build affinity matrix
    kwargs_W = {"metric": "euclidean", "neighbor_mode": "knn", "weight_mode": "heat_kernel", "k": 5, 't': 1}
    W = construct_W.construct_W(X, **kwargs_W)

    # Obtain the scores of all features
    scores = lap_score.lap_score(X, W=W)

    # Sort each column by descending order (higher the score, more important the feature)
    sorted_idx = feat_ranking(scores, 'descend')

    # Selected features
    return get_features(scores, sorted_idx)

def spectral(X, y=None):
    """
    Spectral: similarity based feature selection

    Arguments:
    X -- training set
    y -- actual class labels (only for compatibility with the main code)
    
    Returns:
    indexes of selected features
    """

    # Obtain the scores of all features
    scores = SPEC.spec(X)
    
    # Sort each column by descending order (higher the score, more important the feature)
    sorted_idx = feat_ranking(scores, 'descend')
    
    # Selected features
    return get_features(scores, sorted_idx)

def rrfs_mm(X, y=None, m=None, ms=0.7):
    """
    Relevance-Redundancy Feature Selection
    Algorithm 6 of the master's thesis (Subsection 4.2.2), with mean-median as a measure of relevance

    Arguments:
    X -- training set
    y -- actual class labels (only for compatibility with the main code)
    m -- maximum number of features to keep after feature selection
    ms -- maximum similarity between remaining features (0 < ms < 1)
    
    Returns:
    indexes of selected features
    """

    # Compute the relevance for each feature, using the mean-median measure (mm)
    mm = np.abs(np.mean(X, axis=0) - np.median(X, axis=0))
    #print("mm.shape", mm.shape)
    #print("mm", mm)
    
    # Selected features
    return rrfs(X, mm, m, ms)

#.............................. SUPERVISED METHODS ............................#

def fisher_ratio(X, y):
    """
    Fisher Score: similarity based feature selection

    Arguments:
    X -- training set
    y -- actual class labels
    
    Returns:
    indexes of selected features
    """
    
    # Obtain the scores of all features (fisher_score = 1/lap_score - 1)
    scores = fisher_score.fisher_score(X, y)

    # Sort each column by descending order (higher the score, more important the feature)
    sorted_idx = feat_ranking(scores, 'descend')

    # Selected features
    return get_features(scores, sorted_idx)

def rrfs_fisher_ratio(X, y, m=None, ms=0.7):
    """
    Relevance-Redundancy Feature Selection
    Algorithm 6 of the master's thesis (Subsection 4.2.2), with fisher ratio as a measure of relevance

    Arguments:
    X -- training set
    y -- actual class labels
    m -- maximum number of features to keep after feature selection
    ms -- maximum similarity between remaining features (0 < ms < 1)
    
    Returns:
    indexes of selected features
    """

    # Count number of occurrences of each class
    unique_c, n_occurrences_c = np.unique(y, return_counts=True)
    n_occ = n_occurrences_c[:, None]

    # Compute instance mean and variance, which has the same class value, for each feature 
    X_mean = [0]*len(unique_c)
    X_var = [0]*len(unique_c)
    for i in range(len(unique_c)):
        y_idx = np.where(y == unique_c[i])[0]
        X_mean[i] = np.mean(X[y_idx], axis=0)
        X_var[i] = np.var(X[y_idx], axis=0)

    # Compute the relevance for each feature, using the fisher ratio measure (fr)
    fr = np.sum(n_occ * ((X_mean - np.mean(X, axis=0)) ** 2), axis=0) / np.sum(n_occ * X_var, axis=0)
    #print("fr.shape", fr.shape)
    #print("fr", fr)
    
    # Selected features
    return rrfs(X, fr, m, ms)

#.................................... OTHERS ..................................#

def rrfs(X, relevance, m, ms):
    """
    Relevance-Redundancy Feature Selection
    Algorithm 6 of the master's thesis (Subsection 4.2.2)

    Arguments:
    X -- training set
    relevance -- relevance for each feature
    m -- maximum number of features to keep after feature selection
    ms -- maximum similarity between remaining features (0 < ms < 1)
    
    Returns:
    indexes of selected features
    """

    # Dimension
    n, d = X.shape

    if not m:
        m = d

    # Sort each column by descending order (higher the mm, more important the feature)
    sorted_idx = feat_ranking(relevance, 'descend')

    # Keep the the first feature (most relevant)
    first_feat = sorted_idx[0]
    max_features = min(m, d)
    feat_keep = np.array([first_feat] + [0]*(max_features - 1))

    f1 = X[:, sorted_idx[first_feat]]
    j = 1

    # Remove redundancy
    for i in range(1, max_features):
        # Next feature
        f2 = X[:, sorted_idx[i]]

        # Compute similarity among features
        s = np.abs(np.sum(f1 * f2) / ((np.sqrt(np.sum(f1 ** 2))) * (np.sqrt(np.sum(f2 ** 2)))))
        
        # Features to keep
        if s < ms:
            feat_keep[j] = sorted_idx[i]
            j = j + 1
            f1 = X[:, sorted_idx[i]]
            
        if j == m:
            break

    # Selected features
    feat_keep = feat_keep[0:j]
    return feat_keep

def feat_ranking(scores, order='descend'):
    """
    Sort each column

    Arguments:
    scores -- scores of all features
    order -- ranking order (descending or ascending)
    
    Returns:
    sorted index
    """

    if order == 'ascend':
        return np.argsort(scores, 0)

    # Default: descending order
    return np.argsort(scores, 0)[::-1]

def get_features(scores, sorted_idx):
    """
    Get features with the most accumulated relevance (90%)

    Arguments:
    scores -- scores of all features
    sorted_idexes -- sorted indexes of all features
    
    Returns:
    indexes of selected features
    """
    
    # Normalized accumulated relevance
    norm_accum = acumulated_relevance(scores, sorted_idx)

    # Indexes of the selected features in the norm_accum
    idx_fea = np.where(norm_accum < 0.9)[0]

    # Real indexes of the selected features (values from sorted_idx)
    return sorted_idx[idx_fea]

def acumulated_relevance(scores, sorted_idexes):
    """
    Normalize the data between the range 0 and 1

    Arguments:
    scores -- scores of all features
    sorted_idexes -- sorted indexes of all features
    
    Returns:
    data normalized with accumulated relevance
    """

    acumulated_scores = [0]*len(scores)
    acumulated_score = 0
    for idx, value in enumerate(sorted_idexes):
        acumulated_score += scores[value]
        acumulated_scores[idx] = (acumulated_score)
    return normalize(acumulated_scores)

def normalize(X):
    """
    Normalize the data between the range 0 and 1

    Arguments:
    X -- input data
    
    Returns:
    data normalized
    """
    
    if(np.max(X) == np.min(X)):
        return np.array([1/len(X)]*len(X))
    return (X - np.min(X)) / (np.max(X) - np.min(X))

#==============================================================================#
#==============================================================================#
# #                          GLOBAL VARIABLES                                # #
#==============================================================================#
#==============================================================================#

# Dimensionality Reduction Test 1: data normalized
techniques = [(laplacian_score, ()),
              (spectral, ()),
              (rrfs_mm, ()),
              (fisher_ratio, ()),
              (rrfs_fisher_ratio, ())]

# To save the evaluation metrics
eval_fs = "./files/results/pipeline/dimensionality_reduction/" #test1/

# Feature selection techniques to be applied for each dataset (data normalized)
# Note: to be used only on the data_pipeline.py
pipeline_techniques = {"Breast" : [(rrfs_fisher_ratio, ())],
                       "CNS" : [(spectral, ())],
                       "Colon" : [(laplacian_score, ())],
                       "Leukemia" : [(laplacian_score, ())],
                       "Leukemia_3c" : [(rrfs_fisher_ratio, ())],
                       "Leukemia_4c" : [(rrfs_fisher_ratio, ())],
                       "Lung" : [(fisher_ratio, ())],
                       "Lymphoma" : [(laplacian_score, ())],
                       "MLL" : [(rrfs_mm, ())],
                       "Ovarian" : [(rrfs_fisher_ratio, ())],
                       "SRBCT" : [(spectral, ())]}

#==============================================================================#
#==============================================================================#
# #                                MAIN                                      # #
#==============================================================================#
#==============================================================================#

if __name__ == "__main__":
    # Load data
    files_names = load_data.data_files_names
    datas, names = load_data.apply(files_names)

    # True class label
    y_trues = [0]*len(datas)
    
    # Predicted class
    y_preds = [0]*len(datas)

    # Selected features
    features = [0]*len(datas)

    # Counter
    count = [0]*len(datas)

    # Go through all datasets
    for d, (data, data_name) in enumerate(zip(datas, names)):
        print(">> Dataset: " + str(data_name) + " <<\n")
        
        # Preprocessing the data
        X, y = pre_processing.apply(data, data_name)

        # Counter
        count[d] = np.array([{index : 0 for index in range(len(X[0]))}]*len(techniques))

        # True classes
        y_trues[d] = y
        
        # Choose split data technique (Leave-one-out cross validation) and get indexes
        split_indexes = split_data.apply(split_data.techniques[0][0], *split_data.techniques[0][1])

        # Predicted class
        y_preds[d] = np.array([[[None]*split_indexes.get_n_splits(X)]*len(data_classification.techniques)]*len(techniques))

        # Selected features
        features[d] = np.array([[None]*split_indexes.get_n_splits(X)]*len(techniques))
        
        # Do leave-one-out cross validation
        for s, (train_index, test_index) in enumerate(split_indexes.split(X,y)):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Create normalizer and fit it
            normalizer = data_representation.normalize(x_train)
            
            # Normalize the data
            n_x_train = normalizer.transform(x_train)
            n_x_test = normalizer.transform(x_test)

            # Go through all feature selection techniques
            for fs, (fs_method, fs_args) in enumerate(techniques):
                # Select features index
                sel_feat = apply(fs_method, n_x_train, y_train, *fs_args)
                features[d][fs][s] = sel_feat
                print("s:", s, "\n", "len(sel_feat):", len(sel_feat), "\n", "sel_feat:", sel_feat, "\n")

                # Datasets with the selected features
                fs_x_train = n_x_train[:, sel_feat]
                fs_x_test = n_x_test[:, sel_feat]

                # Count selected features
                for i in sel_feat:
                    count[d][fs][i] += 1

                # Go through all classification techniques
                for c, (classif_method, classif_args) in enumerate(data_classification.techniques):
                    y_pred = data_classification.apply(classif_method, fs_x_train, y_train, fs_x_test, *classif_args)
                    y_preds[d][fs][c][s] = y_pred

        # Uncomment to only do the first dataset
        #break

    #............................................................................#
    # Save in pickle File
    pickle.dump(y_trues, open(eval_fs + "y_trues.p", "wb"))
    p_y_trues = pickle.load(open(eval_fs + "y_trues.p", "rb"))
    
    pickle.dump(y_preds, open(eval_fs + "y_preds.p", "wb"))
    p_y_preds = pickle.load(open(eval_fs + "y_preds.p", "rb"))

    pickle.dump(features, open(eval_fs + "features.p", "wb"))
    p_features = pickle.load(open(eval_fs + "features.p", "rb"))

    pickle.dump(count, open(eval_fs + "count.p", "wb"))
    p_count = pickle.load(open(eval_fs + "count.p", "rb"))
    
    #............................................................................#
    #.......................... Estimate and Visualize ..........................#
    # Save in text File
    with open(eval_fs + "dimensionality_reduction.txt", "w") as file:

        # Evaluation metrics
        evaluations = [0]*len(datas)

        # Go through all datasets 
        for d, data_name in enumerate(names):
            print(">> Dataset: " + str(data_name) + " <<\n")
            file.write("#............................................................................#\n\n")
            file.write(">> Dataset: " + str(data_name) + "\n")

            # Evaluation metrics
            evaluations[d] = np.array([[None]*len(data_classification.techniques)]*len(techniques))
            
            # Go through all feature selection techniques
            for fs, (fs_method, _) in enumerate(techniques):
                print("- Dimensionality Reduction Technique: " + str(fs_method.__name__) + " <<\n")
                file.write("- Dimensionality Reduction Technique: " + str(fs_method.__name__) + "\n")

                # Go through all classification techniques
                for c, (classif_method, _) in enumerate(data_classification.techniques):
                    print("- Classification Technique: " + str(classif_method.__name__) + "\n")
                    file.write("- Classification Technique: " + str(classif_method.__name__) + "\n")

                    # Confusion Matrix - Algorithm 8 of the master's thesis (Subsection 4.2.4)
                    # Rows: contains the true class | Columns: contains the predicted class
                    confusion_matrix, mapping = evaluation_metrics.initiate_confusion_matrix(y_trues[d])
                    for i in range(len(y_preds[d][fs][c])):
                        confusion_matrix[mapping[y_trues[d][i]]][mapping[y_preds[d][fs][c][i]]] +=1

                    name = "dataset: " + data_name + " | feature selection method: " + fs_method.__name__ + " | classification method: " + classif_method.__name__
                    evaluation = evaluation_metrics.calculate(name, confusion_matrix)
                    evaluations[d][fs][c] = evaluation

                    # Save evaluation metrics in a text file
                    file.write(". Accuracy " + str(classif_method.__name__) + ": " + str(evaluation['accuracy']) + "\n")
                    file.write(". Error " + str(classif_method.__name__) + ": " + str(evaluation['error']) + "\n")
                    file.write(". Precision " + str(classif_method.__name__) + ": " + str(evaluation['precision']) + "\n")
                    file.write(". F-measure " + str(classif_method.__name__) + ": " + str(evaluation['f_measure']) + "\n")
                    file.write(". Confusion Matrix " + str(classif_method.__name__) + ":\n " + str(evaluation['confusion_matrix']) + "\n")
                    file.write(". tp, fp, tn, fn " + str(classif_method.__name__) + ": " + str(evaluation['tp']) + " " + str(evaluation['fp']) + " " + str(evaluation['tn']) + " " + str(evaluation['fn']) + "\n")
                    file.write(". False Negative Rate (miss rate) " + str(classif_method.__name__) + ": " + str(evaluation['fnr']) + "\n")
                    file.write(". True negative rate (specificity) " + str(classif_method.__name__) + ": " + str(evaluation['tnr']) + "\n")
                    file.write(". True positive rate (recall) " + str(classif_method.__name__) + ": " + str(evaluation['tpr']) + "\n")
                    file.write(". False positive rate (false alarm rate) " + str(classif_method.__name__) + ": " + str(evaluation['fpr']) + "\n\n")
                    
                file.write("\n")        
            file.write("\n")
        file.write("#............................................................................#\n")
        file.close()

    #............................................................................#
    # Save in pickle File
    pickle.dump(evaluations, open(eval_fs + "evaluations.p", "wb"))
    p_evaluations = pickle.load(open(eval_fs + "evaluations.p", "rb"))

