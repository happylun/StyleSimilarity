# Elements of Style: Learning Perceptual Shape Style Similarity

[Project Page](http://people.cs.umass.edu/~zlun/papers/StyleSimilarity)

## Introduction

This archive contains Matlab codes for the **learning** part of the Style Similarity project. Only Matlab R2013b or above are tested and guaranteed to be able to run the code correctly.

## Before Running

- This program is only one part of the entire pipeline. Before running the learning code, you will need to first run the C++ program in folder `../Metric/` to get the similarity metrics between shape pairs. Check the readme file in the C++ project folder for more details
- After running the C++ program on a data set, you should get sub-folders `saliency` and `triplet` inside the data set folder. Make sure those sub-folders are not empty before running the learning code

## How to Run

- run script `loadData.m` to load the metric data. Make sure you have set the input folder correctly in this script (relative to current working directory)
- run script `learn.m` to train and test the learning algorithm. You can set the number of folds for cross-validation testing in this script (we suggest 10-fold cross-validation for the large data sets)