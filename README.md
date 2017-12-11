# A (performance) comparison of different implementations of the Householder way of QR factorization

## What is this?

This is a Jupyter notebook about QR factorization with the Householder algorithm.  
QR factorization means that given a matrix with dimensions m x n (where m >= n) we want to compute matrices Q and R such that A = QR.
Here Q is a unitary matrix and R is an upper triangular matrix.

## Why this?

I am a first year masters student in computational engineering and I like Python. I have no idea about C++.
These are my first steps with C++ and xtensor.  

The notebook contains a Python and Numpy implementation,  
followerd by a Numba jit-compiled version of the same algorithm,
and ultimately it should contain also a C++ implementation with Python bindings.

For the latter I chose to use
- [pybind11](http://pybind11.readthedocs.io/en/stable/)
- [xtensor](http://xtensor.readthedocs.io/en/stable/)
- [xtensor-python](http://xtensor-python.readthedocs.io/en/stable/)
- [xtensor-blas](http://xtensor-blas.readthedocs.io/en/stable/)
