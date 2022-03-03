

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

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np
import pickle

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

def apply(classif_method, x_train, y_train, x_test, *classif_args):
    """
    Apply the classification technique to the data
    Algorithm 6 of the master's thesis (Subsection 4.2.3)

    Arguments:
    classif_method -- classification technique
    x_train -- input data to fit the model
    y_train -- actual class labels to fit the model
    x_test -- input data to test the model
    classif_args -- classification technique parameters
    
    Returns:
    predicted class label
    """

    print("- Classification Technique: " + str(classif_method.__name__) + "\n")

    # Create classifier
    classifier = classif_method(*classif_args)

    # Fit the model
    classifier.fit(x_train, y_train)

    # Predict
    y_pred = classifier.predict(x_test)
    return y_pred[0]

#.............................. SUPERVISED METHODS ............................#

def svm(C, kernel):
    """
    Support Vector Machines

    Arguments:
    C -- regularization parameter
    kernel -- kernel type to be used in the classifier

    Returns:
    classifier
    """
    return SVC(C=C, kernel=kernel)

def decison_tree(criterion, max_depth, random_state):
    """
    Decision Trees

    Arguments:
    criterion -- function to measure the quality of a split
    max_depth -- maximum depth of the tree
    random_state -- used by a random generator to control the randomness of the classifier

    Returns:
    classifier
    """
    return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, random_state=random_state)

#==============================================================================#
#==============================================================================#
# #                          GLOBAL VARIABLES                                # #
#==============================================================================#
#==============================================================================#

### Classification techniques to be applied (best overall)
techniques = [(svm, (1, 'linear',)),
              (decison_tree, ('entropy', 5, 42,))]

### Baseline Test 1 (raw data) and 2 (data normalized) - svm:kernel, dt:criterion and random_state
##techniques = [(svm, (1, 'linear',)),
##              (svm, (1, 'poly',)),
##              (svm, (1, 'rbf',)), # default
##              (svm, (1, 'sigmoid',)),
##              (decison_tree, ('gini', None, 5,)),
##              (decison_tree, ('gini', None, 13,)),
##              (decison_tree, ('gini', None, 29,)),
##              (decison_tree, ('gini', None, 42,)),
##              (decison_tree, ('entropy', None, 5,)),
##              (decison_tree, ('entropy', None, 13,)),
##              (decison_tree, ('entropy', None, 29,)),
##              (decison_tree, ('entropy', None, 42,))]

### Baseline Test 3 (data normalized) - svm:C, dt:max_depth
##techniques = [(svm, (1.5, 'linear',)),
##              (svm, (2, 'linear',)),
##              (svm, (10, 'linear',)),
##              (svm, (100, 'linear',)),
##              (decison_tree, ('entropy', 2, 42,)),
##              (decison_tree, ('entropy', 5, 42,)),
##              (decison_tree, ('entropy', 7, 42,)),
##              (decison_tree, ('entropy', 10, 42,))]

# To save the evaluation metrics
eval_baseline = "./files/results/baseline/" #test3/

# Classification techniques to be applied for each dataset (data normalized)
# Note: to be used only on the data_pipeline.py
pipeline_techniques = {"Breast" : [(svm, (1, 'linear',))],
                       "CNS" : [(decison_tree, ('entropy', 5, 42,))],
                       "Colon" : [(decison_tree, ('entropy', None, 5,))],
                       "Leukemia" : [(svm, (1, 'linear',))],
                       "Leukemia_3c" : [(svm, (1, 'linear',))],
                       "Leukemia_4c" : [(svm, (1, 'linear',))],
                       "Lung" : [(svm, (1, 'linear',))],
                       "Lymphoma" : [(svm, (1, 'linear',))],
                       "MLL" : [(svm, (1, 'linear',))],
                       "Ovarian" : [(svm, (1, 'linear',))],
                       "SRBCT" : [(svm, (1, 'linear',))]}

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
        y_preds[d] = np.array([[None]*split_indexes.get_n_splits(X)]*len(techniques))

        # Do leave-one-out cross validation
        for s, (train_index, test_index) in enumerate(split_indexes.split(X,y)):
            x_train, x_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Create normalizer and fit it
            normalizer = data_representation.normalize(x_train)
            
            # Normalize the data
            n_x_train = normalizer.transform(x_train)
            n_x_test = normalizer.transform(x_test)
            
            # Go through all classification techniques
            for c, (classif_method, classif_args) in enumerate(techniques):
                #y_pred = apply(classif_method, x_train, y_train, x_test, *classif_args)
                y_pred = apply(classif_method, n_x_train, y_train, n_x_test, *classif_args)
                y_preds[d][c][s] = y_pred

        # Uncomment to only do the first dataset
        #break

    #............................................................................#
    # Save in pickle File
    pickle.dump(y_trues, open(eval_baseline + "y_trues.p", "wb"))
    p_y_trues = pickle.load(open(eval_baseline + "y_trues.p", "rb"))
    
    pickle.dump(y_preds, open(eval_baseline + "y_preds.p", "wb"))
    p_y_preds = pickle.load(open(eval_baseline + "y_preds.p", "rb"))
    
    #............................................................................#
    #.......................... Estimate and Visualize ..........................#
    # Save in text File
    with open(eval_baseline + "baseline.txt", "w") as file:

        # Evaluation metrics
        evaluations = [0]*len(datas)

        # Go through all datasets
        for d, data_name in enumerate(names):
            print(">> Dataset: " + str(data_name) + " <<\n")
            file.write("#............................................................................#\n\n")
            file.write(">> Dataset: " + str(data_name) + "\n")

            # Evaluation metrics
            evaluations[d] = np.array([None]*len(techniques))

            # Go through all classification techniques
            for c, (classif_method, _) in enumerate(techniques):
                print("- Classification Technique: " + str(classif_method.__name__) + "\n")
                file.write("- Classification Technique: " + str(classif_method.__name__) + "\n")
                
                # Confusion Matrix - Algorithm 8 of the master's thesis (Subsection 4.2.4)
                # Rows: contains the true class | Columns: contains the predicted class
                confusion_matrix, mapping = evaluation_metrics.initiate_confusion_matrix(y_trues[d])
                for i in range(len(y_preds[d][c])):
                    confusion_matrix[mapping[y_trues[d][i]]][mapping[y_preds[d][c][i]]] +=1

                name = "dataset: " + data_name + " | classification method: " + classif_method.__name__
                evaluation = evaluation_metrics.calculate(name, confusion_matrix)
                evaluations[d][c] = evaluation

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
        file.write("#............................................................................#\n")
        file.close()

    #............................................................................#
    # Save in pickle File
    pickle.dump(evaluations, open(eval_baseline + "evaluations.p", "wb"))
    p_evaluations = pickle.load(open(eval_baseline + "evaluations.p", "rb"))

