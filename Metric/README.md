# Elements of Style: Learning Perceptual Shape Style Similarity

[Project Page](http://people.cs.umass.edu/~zlun/papers/StyleSimilarity)

## Introduction

This archive contains C++ codes for the **metric** part of the Style Similarity project. Visual Studio 2012 or above is required to compile the code. 

## How to Configure ##

1. Open Visual Studio solution `build/Style.sln`
2. In Visual Studio, click `VIEW` > `Property Manager`
3. In `Property Manager` panel, expand project `StyleSimilarity` and open property sheet `Macros` under any configuration (e.g. `Debug | Win32`)
4. In `Common Properties` > `User Macros`, configure all macros according to your own environment
5. There's no need to configure the property sheet under other configurations (e.g. `Release | x64`). They share the same property sheet file


## Third-party Libraries ##

1. All third-party library header files and pre-built binary files are included in folder `3rdparty` except for the [Boost](http://www.boost.org/) library. You will need to set up the Boost library in your development environment and configure the library paths using the Visual Studio property sheet as instructed above
2. This project requires OpenGL. You will need an OpenGL capable graphics card and development environment
3. All third-party libraries used in this project are listed below:
	- [ARPACK 2.1](http://www.caam.rice.edu/software/ARPACK): solving eigenvalue problem
	- [Boost 1.55.0](http://sourceforge.net/projects/boost/files/boost-binaries): general purpose C++ library
	- [CML 1.0.3](http://cmldev.net): vector, matrix and quaternion operations  
	- [Eigen 3.2.0](http://eigen.tuxfamily.org): solving linear system
	- [FFTW 3.3.4](http://fftw.org): computing discrete Fourier transform
	- [FLANN 1.8.4](http://www.cs.ubc.ca/research/flann): nearest neighbors search in high dimension
	- [GLEW 1.11.0](http://glew.sourceforge.net): OpenGL Extension Wrangler Library
	- [maxflow 3.01](http://vision.csd.uwo.ca/code): solving max-flow/min-cut problem
	- [Thea](https://github.com/sidch/Thea): using the Kd tree implementation (requires Boost library)

## How to Run

- After compiling the project, the binary executables are inside folder `output`
- You can either run the program with the accompanying demo data set (`../../Data/demo/`) or download the large data sets from the [project page](http://people.cs.umass.edu/~zlun/papers/StyleSimilarity) and run the program on a specific data set of shapes
- To run the program on a specific data set, simply run the executable using the `*.cfg` configuration files as input argument. The program can take several configuration files as arguments
- Make sure the data set root folder is set correctly (relative to current working directory) in the configuration file before running
- Taking the `demo` data set for example: assuming the current working directory is inside `Data` folder, you can run the program with this command

	```
    	StylySimilarity.exe params.cfg demo/demo.cfg
	```


## Other Notes

- For demonstration purpose, the pipeline is not parallelized. You can easily parallelize each step of the pipeline with [OpenMP](http://openmp.org). Check the main routine in file `src/StyleSimilarity/IO/Run/DemoIO.cpp` for more details
- This program is only one part of the entire pipeline. After running the program on a data set, you will need to run the Matlab code in folder `../Learning/` to get the learning result. Check the readme file in the Matlab project folder for more details