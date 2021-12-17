# MLOC: Machine Learning Optimizer using Cascading

> This repository contains code for the final project for Deep Learning Practical Systems (COMS 6998). 


Submission By:
1. Gaurav Sinha (gs3157)
2. Gursifath Bhasin (gb2760) 

## Description of the Project
Machine learning inference is an increasingly important workload in modern data centers. The cost of computing features, however, is often a bottleneck in most ML production systems. Our goal through this project is to decrease the inference time without compromising on the performance of the model.

Since ML models are amenable to approximation, they can often classify data inputs using only a subset of these features. Our project aims to quicken the ML inference process by exploiting this property of ML models via end-to-end cascading, i.e., classifying data inputs using only necessary features. 

Approach we followed:
1. We first visualize the data and then group the data based on the correlated features.
2. We dump these correlated features into a Redis instance as a .pkl file for rapid retrieval.
3. For training:

      a. we split the data into train and test sets.
      
      b. compute the feature groups (the features which are generated from the same upstream operators are not computationally independent and must be grouped together) 
      
      c. for each feature group, we calculate: **permutation importance** (measure of the value of this feature group to the model’s predictions), and **computational cost** (amount of time it takes to compute the feature group on a sample of the training set)
          
      d. select a subset of features that have minimal cost but high prediction accuracy.
      
      e. train an approximate model from a selected set of feature groups.
      
      f. choose a cascade threshold for an approximate model, which is lowest value for which the prediction of the approximate model will be considered. 
   

## Description of the Repository

Our code is mainly divided into two packages: 
`Graph` and `Runtime` found under the `/src` folder.

`Graph`: contains the utility functions for defining the abstract syntax tree using Python

`Runtime`: contains  functionalities such as: 

    1. graphbuilder.py: builds graph for feature computation into sets
    
    2. timer.py:  takes care of all the time calculation
    
    3. cascade_construct.py: constructs cascades and holds the cutoffs and calculate feature perf.
    
    4. cascadepredict.py:  predicts the cascades and calculates the indices of approximation and combines prediction
    
    5. executor.py: has decorator functions that helps to run approximated models.
    
    
The `/preprocess_scripts` folder consists of:

    1. `/benchmark_ex` folder where we define the training and testing pipelines.
    
    2. `/data` folder which contains our preprocessed data stored in .csv format. The dataset we employed for this project is the ![alt text](https://www.kaggle.com/c/kkbox-music-recommendation-challenge)
    
The `/notebooks` folder consists of the following .ipynb notebooks:

    1. `catboost_music.ipynb`: 
    
    2. `lightgbm_music.ipynb`:
    
    3. `music_recommendation_Performance.ipynb`: 
    
    4. `music_recommendation_feature_analysis.ipynb`: 


## Commands to execute the code

## Results

