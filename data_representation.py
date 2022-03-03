

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

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import KBinsDiscretizer
from mdlp.discretization import MDLP
from scipy.cluster.vq import vq, kmeans, whiten
import pandas as pd
import numpy as np
import pickle

import data_classification
import evaluation_metrics
import pre_processing
import load_data
import split_data

#==============================================================================#
#==============================================================================#
# #                               METHODS                                    # #
#==============================================================================#
#==============================================================================#
#.................................. MAIN METHODS ...............................#

def apply(discret_method, X, y, x_test, *discret_args):
    """
    Apply the discretization technique to the data
    Algorithm 2 of the master's thesis (Subsection 4.2.1)

    Arguments:
    discret_method -- discretization technique
    X -- training set
    y -- actual class labels
    x_test -- test set
    discret_args -- discretization technique parameters
    
    Returns:
    discretized training and test sets
    """

    print("- Data Representation Technique: " + str(discret_method.__name__) + " <<\n")

    q_bits = None
    if str(discret_method.__name__) == "efb" or str(discret_method.__name__) == "mdl":
        discretizer = discret_method(X, y, x_test, *discret_args)
        d_x_train = discretizer.transform(X)
        d_x_test = discretizer.transform(x_test)
    elif str(discret_method.__name__) == "u_lgb1" or str(discret_method.__name__) == "r_lgb":
        d_x_train, d_x_test, q_bits = discret_method(X, y, x_test, *discret_args)
    return d_x_train, d_x_test, q_bits

def normalize(X):
    """
    Normalize the data, column wise, between the range 0 and 1

    Arguments:
    X -- training set
    
    Returns:
    normalizer
    """
    normalizer = MinMaxScaler()
    return normalizer.fit(X)

#............................ UNSUPERVISED METHODS ............................#

def efb(X, y=None, x_test=None, n_bins=3):
    """
    Equal Frequency Binning

    Arguments:
    X -- training set
    y -- actual class labels (only for compatibility with the main code)
    x_test -- test set (only for compatibility with the main code)
    n_bins -- number of bins
    
    Returns:
    discretizer
    """
    
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
    return discretizer.fit(X)

def u_lgb1(X, y=None, x_test=None, q=3):
    """
    Linde-Buzo-Gray 1
    Algorithm 3 of the master's thesis (Subsection 4.2.1)

    Arguments:
    X -- training set
    y -- actual class labels (only for compatibility with the main code)
    x_test -- test set (the default value is set to None only for compatibility with the main code. We always have a x_test to be discretized)
    q -- maximum number of bits per feature
    
    Returns:
    X_discretized -- discretized training set
    x_test_discretized -- discretized test set
    """
    
    # Dimension
    n, d = X.shape

    # Number of quantization bits applied for each feature
    q_bits = q * np.ones((d))

    X_discretized = np.zeros(X.shape)
    x_test_discretized = np.zeros(x_test.shape)

    for i in range(0, d):
        # Standard deviation normalization, kmeans converge more easily
        f = np.concatenate((X[:, i], x_test[:, i]))
        f = whiten(f)
        
        f_train = np.array(f[:-1])
        f_test = np.array(f[-1])

        range_f = max(f_train) - min(f_train)
        
        # 2^j number of bins
        for j in range(1, q + 1):

            # Apply kmeans (100 iter)
            codebook, distortion = kmeans(f_train, 2**j, 100)
            
            if distortion < 0.05*range_f or j == q:
                q_bits[i] = j
                
                code_idx, _ =  vq(f_train, codebook)
                X_discretized[:, i] = code_idx

                code_idx, _ =  vq(f_test, codebook)
                x_test_discretized[:, i] = code_idx

                # Move on to the next feature
                break
    
    return X_discretized, x_test_discretized, q_bits

#.............................. SUPERVISED METHODS ............................#

def mdl(X, y, x_test=None):
    """
    Minimum Description Length Principl

    Arguments:
    X -- training set
    y -- actual class labels
    x_test -- test set (only for compatibility with the main code)
    
    Returns:
    discretizer
    """

    discretizer = MDLP(np.arange(X.shape[1]))
    return discretizer.fit(X, y)

def r_lgb(X, y, x_test, q, delta):
    """
    Relevance Linde-Buzo-Gray
    Algorithm 4 of the master's thesis (Subsection 4.2.1)

    Arguments:
    X -- training set
    y -- actual class labels
    x_test -- test set
    q -- maximum number of bits per feature
    delta -- 
    
    Returns:
    X_discretized -- discretized training set
    x_test_discretized -- discretized test set
    """

    # Dimension
    n, d = X.shape
    n_test, d_test = x_test.shape

    X_discretized = np.zeros(X.shape)
    x_test_discretized = np.zeros(x_test.shape)

    # Number of quantization bits applied for each feature
    q_bits = q * np.ones((d))
    
    # Count number of occurrences of each class
    unique_c, n_occurrences_c = np.unique(y, return_counts=True)
    n_occ = n_occurrences_c[:, None]

    for i in range(0, d):
        # Standard deviation normalization, kmeans converge more easily
        f = np.concatenate((X[:, i], x_test[:, i]))
        f = whiten(f)
        
        f_train = np.array(f[:-1])
        f_test = np.array(f[-1])

        prev_feature_interaction = 0

        # 2^j number of bins
        for j in range(1, q + 1):

            # Apply kmeans (100 iter)
            codebook, _ = kmeans(f_train, 2**j, 100)
            code_idx, _ =  vq(f_train, codebook)

            # Compute instance mean and variance, which has the same class value, for each feature 
            X_mean = [0]*len(unique_c)
            X_var = [0]*len(unique_c)
            for c in range(len(unique_c)):
                y_idx = np.where(y == unique_c[c])[0]
                X_mean[c] = np.mean(code_idx[y_idx])
                X_var[c] = np.var(code_idx[y_idx])
            
            # Compute the relevance for each feature, using the fisher ratio measure (fr)
            fr = np.sum(n_occ * ((X_mean - np.mean(code_idx, axis=0)) ** 2)) / np.sum(n_occ * X_var)
            #print("fr", fr)
            
            if fr - prev_feature_interaction > delta or j == q:
                q_bits[i] = j

                # Quantize train set
                X_discretized[:, i] = code_idx

                # Quantize test set
                code_idx, _ =  vq(f_test, codebook)
                x_test_discretized[:, i] = code_idx

                # Move on to the next feature
                break

            prev_feature_interaction = fr
    
    return X_discretized, x_test_discretized, q_bits

#==============================================================================#
#==============================================================================#
# #                          GLOBAL VARIABLES                                # #
#==============================================================================#
#==============================================================================#

# Discretization techniques to be applied
techniques = [(efb, (5,)),
              (mdl, ())]

### Data Representation Test 1 (data normalized - efb:n_bins, mdl:default vaulues)
##techniques = [(efb, (2,)),
##              (efb, (3,)),
##              (efb, (4,)),
##              (efb, (5,)),
##              (efb, (6,)),
##              (efb, (7,)),
##              (mdl, ())]

# To save the evaluation metrics
eval_disc = "./files/results/pipeline/data_representation/" # test2/

# Data representation techniques to be applied for each dataset (data normalized)
# Note: to be used only on the data_pipeline.py - test1
##pipeline_techniques = {"Breast" : [(efb, (6,))],
##                       "CNS" : [(efb, (5,))],
##                       "Colon" : [(mdl, ())],
##                       "Leukemia" : [(efb, (2,))],
##                       "Leukemia_3c" : [(efb, (2,))],
##                       "Leukemia_4c" : [(efb, (3,))],
##                       "Lung" : [(efb, (5,))],
##                       "Lymphoma" : [(efb, (2,))],
##                       "MLL" : [(efb, (3,))],
##                       "Ovarian" : [(efb, (3,))],
##                       "SRBCT" : [(efb, (2,))]}

# Data representation techniques to be applied for each dataset (data normalized)
# Note: to be used only on the data_pipeline.py - test3
pipeline_techniques = {"Breast" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "CNS" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "Colon" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "Leukemia" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "Leukemia_3c" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "Leukemia_4c" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "Lung" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "Lymphoma" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "MLL" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "Ovarian" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))],
                       "SRBCT" : [(u_lgb1, (4,)), (r_lgb, (4, 0.1,))]}

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
    
    # Go through all datasets
    for d, (data, data_name) in enumerate(zip(datas, names)):
        print(">> Dataset: " + str(data_name) + " <<\n")
        
        # Preprocessing the data
        X, y = pre_processing.apply(data, data_name)

        # True classes
        y_trues[d] = y
        
        # Choose split data technique (Leave-one-out cross validation) and get indexes
        split_indexes = split_data.apply(split_data.techniques[0][0], *split_data.techniques[0][1])

        # Predicted class
        y_preds[d] = np.array([[[None]*split_indexes.get_n_splits(X)]*len(data_classification.techniques)]*len(techniques))

        # Do leave-one-out cross validation
        for s, (train_index, test_index) in enumerate(split_indexes.split(X,y)):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Create normalizer and fit it
            normalizer = normalize(x_train)
            
            # Normalize the data
            n_x_train = normalizer.transform(x_train)
            n_x_test = normalizer.transform(x_test)

            # Go through all data representation techniques
            for disc, (discret_method, discret_args) in enumerate(techniques):
                # Discretize
                d_x_train, d_x_test, _ = apply(discret_method, n_x_train, y_train, n_x_test, *discret_args)

                # Go through all classification techniques
                for c, (classif_method, classif_args) in enumerate(data_classification.techniques):                    
                    y_pred = data_classification.apply(classif_method, d_x_train, y_train, d_x_test, *classif_args)
                    y_preds[d][disc][c][s] = y_pred

        # Uncomment to only do the first dataset
        #break

    #............................................................................#
    # Save in pickle File
    pickle.dump(y_trues, open(eval_disc + "y_trues.p", "wb"))
    p_y_trues = pickle.load(open(eval_disc + "y_trues.p", "rb"))
    
    pickle.dump(y_preds, open(eval_disc + "y_preds.p", "wb"))
    p_y_preds = pickle.load(open(eval_disc + "y_preds.p", "rb"))
    
    #............................................................................#
    #.......................... Estimate and Visualize ..........................#
    # Save in text File
    with open(eval_disc + "data_representation.txt", "w") as file:

        # Evaluation metrics
        evaluations = [0]*len(datas)

        # Go through all datasets
        for d, data_name in enumerate(names):
            print(">> Dataset: " + str(data_name) + " <<\n")
            file.write("#............................................................................#\n\n")
            file.write(">> Dataset: " + str(data_name) + "\n")

            # Evaluation metrics
            evaluations[d] = np.array([[None]*len(data_classification.techniques)]*len(techniques))

            # Go through all data representation techniques
            for disc, (discret_method, _) in enumerate(techniques):
                print("- Discretization Technique: " + str(discret_method.__name__) + " <<\n")
                file.write("- Discretization Technique: " + str(discret_method.__name__) + "\n")

                # Go through all classification techniques
                for c, (classif_method, _) in enumerate(data_classification.techniques):
                    print("- Classification Technique: " + str(classif_method.__name__) + "\n")
                    file.write("- Classification Technique: " + str(classif_method.__name__) + "\n")
                    
                    # Confusion Matrix - Algorithm 8 of the master's thesis (Subsection 4.2.4)
                    # Rows: contains the true class | Columns: contains the predicted class
                    confusion_matrix, mapping = evaluation_metrics.initiate_confusion_matrix(y_trues[d])
                    for i in range(len(y_preds[d][disc][c])):
                        confusion_matrix[mapping[y_trues[d][i]]][mapping[y_preds[d][disc][c][i]]] +=1

                    name = "dataset: " + data_name + " | discretization method: " + discret_method.__name__ + " | classification method: " + classif_method.__name__
                    evaluation = evaluation_metrics.calculate(name, confusion_matrix)
                    evaluations[d][disc][c] = evaluation

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
                    file.write("t. False positive rate (false alarm rate) " + str(classif_method.__name__) + ": " + str(evaluation['fpr']) + "\n\n")

                file.write("\n")   
            file.write("\n")
        file.write("#............................................................................#\n")
        file.close()

    #............................................................................#
    # Save in pickle File
    pickle.dump(evaluations, open(eval_disc + "evaluations.p", "wb"))
    p_evaluations = pickle.load(open(eval_disc + "evaluations.p", "rb"))

