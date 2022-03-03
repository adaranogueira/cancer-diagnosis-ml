

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
import numpy as np

import evaluation_metrics
import load_data
import split_data

#==============================================================================#
#==============================================================================#
# #                               METHODS                                    # #
#==============================================================================#
#==============================================================================#
#.................................. MAIN METHOD ...............................#

def apply(data, data_name):
    """
    Pre-processing the data
    Algorithm 1 line 2 of the master's thesis (Section 4.2)

    Arguments:
    data -- input dataset
    data_name -- dataset name
    
    Returns:
    X -- data
    y -- class labels
    """

    print("- Pre-processing: \n")

    # Split the dataset into features and class
    X, y = split_data.features_class_split(data)

    # Transform class labels into integer values
    y = transform_label(y)

    # Deal with missing values
    X, y = deal_missing_values(X, y, data_name)
    
    # Remove constant features
    X = remove_constant_feat(X)
    return X, y

#.................................... OTHERS ..................................#

def transform_label(y):
    """
    Transform class labels into integer values

    Arguments:
    y -- class labels
    
    Returns:
    y -- class labels
    """
    
    if type(y[0]) == str:
        mapping = evaluation_metrics.map_label_index(y)
        y = np.array([mapping[label] for label in y])
    return y

def deal_missing_values(X, y, data_name):
    """
    Deal with missing values, completing it with the most frequent value

    Arguments:
    X -- input data
    y -- class labels
    data_name -- dataset name
    
    Returns:
    X -- data
    y -- class labels
    """
    
    if data_name=="Lymphoma" and np.count_nonzero(sum(X == '?')) is not 0:        
        imp_mean = SimpleImputer(missing_values='?', strategy='most_frequent')
        X = imp_mean.fit_transform(X,y)
        
        # Convert numeric string to float
        X = X.astype(np.float64)
    return X, y

def remove_constant_feat(X):
    """
    Remove constant features

    Arguments:
    X -- input data
    
    Returns:
    X -- data with no constant features
    """

    # Indixes to be deleted
    idx_to_delete = []
    
    # Go through all features
    for i in range(X.shape[1]):
        
        # Find constant features and save its index
        var = np.var(X[:,i])
        if var == 0.0:
            idx_to_delete.append(i)

    # Remove feature i
    if len(idx_to_delete) != 0:
        X = np.delete(X, idx_to_delete, axis=1)

    #print("idx_to_delete", idx_to_delete)
    #print("X.shape", X.shape, "\n")
    
    # Data with no constant features
    return X

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
        
        # Pre-rocessing the data
        X, y = apply(data, data_name)
        
        print("X", X, "\n")
        print("X.shape", X.shape, "\n")
        print("y", y, "\n")
