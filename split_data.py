

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

from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import pandas as pd

import load_data

#==============================================================================#
#==============================================================================#
# #                               METHODS                                    # #
#==============================================================================#
#==============================================================================#

#.......................... FEATURES AND CLASS SPLIT ..........................#

def features_class_split(data):
    """
    Split the microarray dataset into features and class label
    
    Arguments:
    data -- input dataset of type 'pandas.core.frame.DataFrame' and shape
            (input size, number of atributes) = (len(data), features + 1)

    Returns:
    X -- features, all rows and all columns except last column (the class)
    y -- class labels, all rows and just the last column
    """
    
    X = data.values[:,0:-1]
    y = data.values[:,-1]
    return X, y 

#............................ TRAIN AND TEST SPLIT ............................#

def leave_one_out():
    """
    Leave-One-Out cross-validation

    Arguments:

    Returns:
    train and test indices to split the data into train and test sets
    """
    
    return LeaveOneOut()

def k_fold(n_splits, shuffle, random_state=None):
    """
    K-fold cross-validation

    Arguments:
    n_splits -- number of folds
    shuffle -- shuffle the data before splitting it
    random_state -- seed

    Returns:
    train and test indices to split the data into train and test sets
    """
    
    return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

#.................................. MAIN METHOD ...............................#

def apply(split_method, *split_args):
    """
    Apply the split technique to the data
    Algorithm 1 line 3 of the master's thesis (Section 4.2)

    Arguments:
    split_method -- split technique
    X -- input data
    y -- class labels
    split_args -- split technique parameters
    
    Returns:
    splited indexes
    """
    
    return split_method(*split_args)

#==============================================================================#
#==============================================================================#
# #                          GLOBAL VARIABLES                                # #
#==============================================================================#
#==============================================================================#

techniques = [(leave_one_out, ()),
              (k_fold, (10, True, 3))]

#==============================================================================#
#==============================================================================#
# #                                MAIN                                      # #
#==============================================================================#
#==============================================================================#

if __name__ == "__main__":
    # Load data
    files_names = load_data.data_files_names
    datas, names = load_data.apply(files_names)

    # Go through all datasets
    for data, data_name in zip(datas, names):
        print(">> Dataset:", data_name, "<<\n")

        # Split the datataset into features and class label
        X, y = features_class_split(data)
        print("type(X)", type(X))
        print("type(y)", type(y))

        for method, args in techniques:
            print(">> Train and test split technique:", method.__name__, "<<\n")

            # Get indexes
            split_indexes = apply(method, *args)

            # Do cross validation technique
            for train_index, test_index in split_indexes.split(X,y):
                x_train, x_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                print(x_train, '\n', x_test, '\n', y_train, '\n', y_test)
                print(len(x_train), '\n', len(x_test), '\n', len(y_train), '\n', len(y_test))
            break
        break
