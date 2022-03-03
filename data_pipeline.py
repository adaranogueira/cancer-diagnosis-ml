

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
import pickle

import dimensionality_reduction
import data_representation
import data_classification
import evaluation_metrics
import pre_processing
import split_data
import load_data

#==============================================================================#
#==============================================================================#
# #                          GLOBAL VARIABLES                                # #
#==============================================================================#
#==============================================================================#
# To save the evaluation metrics
evaluat = "./files/results/pipeline/joint_disc_fs/test3/1/" # test3/3/

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

# Discretization bits (u_lgb1, r_lgb)
q_bits = [0]*len(datas)

#==============================================================================#
#==============================================================================#
# #                                MAIN                                      # #
#==============================================================================#
#==============================================================================#

# Algorithm 1 of the master's thesis (Section 4.2)

# Go through all datasets
for d, (data, data_name) in enumerate(zip(datas, names)):
    print(">> Dataset: " + str(data_name) + " <<\n")

    # Get techniques for each dataset
    data_representation_techniques = data_representation.pipeline_techniques[data_name]
    dimensionality_reduction_techniques = dimensionality_reduction.pipeline_techniques[data_name]
    data_classification_techniques = data_classification.pipeline_techniques[data_name]

    print(data_representation_techniques)
    print(dimensionality_reduction_techniques)
    print(data_classification_techniques)
    
    # Preprocessing the data
    X, y = pre_processing.apply(data, data_name)

    # Counter
    count[d] = np.array([[{index : 0 for index in range(len(X[0]))}]*len(dimensionality_reduction_techniques)]*len(data_representation_techniques))
    
    # True classes
    y_trues[d] = y
    
    # Choose split data technique (Leave-one-out cross validation) and get indexes
    split_indexes = split_data.apply(split_data.techniques[0][0], *split_data.techniques[0][1])

    # Predicted class
    y_preds[d] = np.array([[[[None]*split_indexes.get_n_splits(X)]*len(data_classification_techniques)]*len(dimensionality_reduction_techniques)]*len(data_representation_techniques))

    # Selected features
    features[d] = np.array([[[None]*split_indexes.get_n_splits(X)]*len(dimensionality_reduction_techniques)]*len(data_representation_techniques))

    # Discretization bits (u_lgb1, r_lgb)
    q_bits[d] = np.array([[None]*split_indexes.get_n_splits(X)]*2)
    
    # Do leave-one-out cross validation
    for s, (train_index, test_index) in enumerate(split_indexes.split(X,y)):
        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Create normalizer and fit it
        normalizer = data_representation.normalize(x_train)
        
        # Normalize the data
        n_x_train = normalizer.transform(x_train)
        n_x_test = normalizer.transform(x_test)

        # Go through all data representation techniques
        for disc, (discret_method, discret_args) in enumerate(data_representation_techniques):
            # Discretize
            d_x_train, d_x_test, bits = data_representation.apply(discret_method, n_x_train, y_train, n_x_test, *discret_args)
            print("d_x_train", d_x_train)
            print("d_x_test", d_x_test)

            # If exists (not None) >> u_lgb1, r_lgb
            if bits is not None:
                q_bits[d][disc][s] = bits
                print("q_bits - unique:", np.unique(bits), "mean:", np.mean(bits))
            
            # Go through all feature selection techniques
            for fs, (fs_method, fs_args) in enumerate(dimensionality_reduction_techniques):
                # Select features index
                sel_feat = dimensionality_reduction.apply(fs_method, d_x_train, y_train, *fs_args)
                features[d][disc][fs][s] = sel_feat
                print("s:", s, "\n", "len(sel_feat):", len(sel_feat), "\n", "sel_feat:", sel_feat, "\n")
                
                # Datasets with the selected features
                fs_x_train = d_x_train[:, sel_feat]
                fs_x_test = d_x_test[:, sel_feat]

                # Count selected features
                for i in sel_feat:
                    count[d][disc][fs][i] += 1

                # Go through all classification techniques
                for c, (classif_method, classif_args) in enumerate(data_classification_techniques):                    
                    y_pred = data_classification.apply(classif_method, fs_x_train, y_train, fs_x_test, *classif_args)
                    y_preds[d][disc][fs][c][s] = y_pred
    
    # Uncomment to only do the first dataset
    #break

#............................................................................#
# Save in pickle File
pickle.dump(y_trues, open(evaluat + "y_trues.p", "wb"))
p_y_trues = pickle.load(open(evaluat + "y_trues.p", "rb"))

pickle.dump(y_preds, open(evaluat + "y_preds.p", "wb"))
p_y_preds = pickle.load(open(evaluat + "y_preds.p", "rb"))

pickle.dump(features, open(evaluat + "features.p", "wb"))
p_features = pickle.load(open(evaluat + "features.p", "rb"))

pickle.dump(count, open(evaluat + "count.p", "wb"))
p_count = pickle.load(open(evaluat + "count.p", "rb"))

#............................................................................#
#.......................... Estimate and Visualize ..........................#
# Save in text File
with open(evaluat + "pipeline.txt", "w") as file:
    # Evaluation metrics
    evaluations = [0]*len(datas)

    # Go through all datasets 
    for d, data_name in enumerate(names):
        print(">> Dataset: " + str(data_name) + " <<\n")
        file.write("#............................................................................#\n\n")
        file.write(">> Dataset: " + str(data_name) + "\n")

        # Get techniques for each dataset
        data_representation_techniques = data_representation.pipeline_techniques[data_name]
        dimensionality_reduction_techniques = dimensionality_reduction.pipeline_techniques[data_name]
        data_classification_techniques = data_classification.pipeline_techniques[data_name]

        # Evaluation metrics
        evaluations[d] = np.array([[[None]*len(data_classification_techniques)]*len(dimensionality_reduction_techniques)]*len(data_representation_techniques))

        # Go through all data representation techniques
        for disc, (discret_method, _) in enumerate(data_representation_techniques):
            print("- Discretization Technique: " + str(discret_method.__name__) + " <<\n")
            file.write("- Discretization Technique: " + str(discret_method.__name__) + "\n")

            # Go through all feature selection techniques
            for fs, (fs_method, _) in enumerate(dimensionality_reduction_techniques):
                print("- Feature Selection Technique: " + str(fs_method.__name__) + " <<\n")
                file.write("- Feature Selection Technique: " + str(fs_method.__name__) + "\n")

                # Go through all classification techniques
                for c, (classif_method, _) in enumerate(data_classification_techniques):
                    print("- Classification Technique: " + str(classif_method.__name__) + "\n")
                    file.write("- Classification Technique: " + str(classif_method.__name__) + "\n")
                    
                    # Confusion Matrix - Algorithm 8 of the master's thesis (Subsection 4.2.4)
                    # Rows: contains the true class | Columns: contains the predicted class
                    confusion_matrix, mapping = evaluation_metrics.initiate_confusion_matrix(y_trues[d])
                    for i in range(len(y_preds[d][disc][fs][c])):
                        confusion_matrix[mapping[y_trues[d][i]]][mapping[y_preds[d][disc][fs][c][i]]] +=1

                    name = "dataset: " + data_name + " | discretization method: " + discret_method.__name__ + " | feature selection method: " + fs_method.__name__ + " | classification method: " + classif_method.__name__
                    evaluation = evaluation_metrics.calculate(name, confusion_matrix)
                    evaluations[d][disc][fs][c] = evaluation

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
        file.write("\n")
    file.write("#............................................................................#\n")
    file.close()

#............................................................................#
# Save in pickle File
pickle.dump(evaluations, open(evaluat + "evaluations.p", "wb"))
p_evaluations = pickle.load(open(evaluat + "evaluations.p", "rb"))

