# Elements of Style: Learning Perceptual Shape Style Similarity

[Project Page](http://people.cs.umass.edu/~zlun/papers/StyleSimilarity)

## Introduction

This archive contains source codes for the Style Similarity project. The main pipeline consists of two main steps: a) computing style similarity metrics, and b) learning style similarities. The folder `Metric` contains C++ code for the **metric** part of the pipeline while the folder `Learning` contains Matlab code for the **learning** part of the pipeline. Check the readme files inside those sub-folders for more details on each part of the pipeline.

## How to Run

- Run the **metric** part first and then the **learning** part. Check the corresponding readme files for instructions on compiling and running the code.
- The pipeline has an iterative procedure which alternates between the **metric** part and the **learning** part. To run the pipeline iteratively, you will need to alternately launch the program for each part manually

## Other Notes

- This code is released under [GPLv3](http://www.gnu.org/licenses/) license
- If you would like to use our code, please cite the following paper:

> Zhaoliang Lun, Evangelos Kalogerakis, Alla Sheffer,
"Elements of Style: Learning Perceptual Shape Style Similarity",
ACM Transactions on Graphics (Proc. ACM SIGGRAPH 2015)

- For any questions or comments, please contact Zhaoliang Lun ([zlun@cs.umass.edu](mailto:zlun@cs.umass.edu))