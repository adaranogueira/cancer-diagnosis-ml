

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

import pandas as pd

#==============================================================================#
#==============================================================================#
# #                               METHODS                                    # #
#==============================================================================#
#==============================================================================#

#................................. READ DATA ..................................#

def get_dataset_name(file_name):
    return file_name.split("/")[-1].split(".")[0]

def get_dataset_names(files_names):
    names = [""]*len(files_names)
    for f in range(len(files_names)):
        names[f] = get_dataset_name(files_names[f])
    return names

def load_dataset(file_name):
    data = pd.read_csv(file_name)
    return data

def load_datasets(files_names):
    datas = [None]*len(files_names)
    for f in range(len(files_names)):
        datas[f] = load_dataset(files_names[f])
    return datas, get_dataset_names(files_names)

#................................... OTHERS ...................................#

def apply(files_names=None):
    if not files_names:
        # Default value (test)
        return load_dataset("./datasets/Colon.csv")
    return load_datasets(files_names)
    
#==============================================================================#
#==============================================================================#
# #                          GLOBAL VARIABLES                                # #
#==============================================================================#
#==============================================================================#
  
data_files_names = ["./datasets/Breast.csv",
                    "./datasets/CNS.csv",
                    "./datasets/Colon.csv",
                    "./datasets/Leukemia.csv",
                    "./datasets/Leukemia_3c.csv",
                    "./datasets/Leukemia_4c.csv",
                    "./datasets/Lung.csv",
                    "./datasets/Lymphoma.csv",
                    "./datasets/MLL.csv",
                    "./datasets/Ovarian.csv",
                    "./datasets/SRBCT.csv"]

### Part 1 - Pipeline test3
##data_files_names = ["./datasets/Breast.csv",
##                    "./datasets/CNS.csv",
##                    "./datasets/Colon.csv"]

### Part 2 - Pipeline test3
##data_files_names = ["./datasets/Leukemia.csv",
##                    "./datasets/Leukemia_3c.csv",
##                    "./datasets/Leukemia_4c.csv"]

### Part 3 - Pipeline test3
##data_files_names = ["./datasets/Lung.csv",
##                    "./datasets/Lymphoma.csv",
##                    "./datasets/MLL.csv"]

### Part 4 - Pipeline test3
##data_files_names = ["./datasets/Ovarian.csv",
##                    "./datasets/SRBCT.csv"]

#==============================================================================#
#==============================================================================#
# #                                MAIN                                      # #
#==============================================================================#
#==============================================================================#

if __name__ == "__main__":

    datas, names = apply(data_files_names)
    for data, data_name in zip(datas, names):
        print(">> Dataset:", data_name, "<<\n")
        print(data.describe().transpose(), "\n")
        print("* * * "*13, "\n")
        break
