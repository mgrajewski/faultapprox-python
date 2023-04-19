# faultapprox-python
This is a python library for detecting and approximation decision lines/faults in 2D and 3D.
We describe the underlying algorithm in "Detecting and approximating decision boundaries in low dimensional spaces", which is available at arXiv.org (http://arxiv.org/abs/2302.08179).

The python packages required are listed in requirements.txt.
There is no explicit documentation yet, but all functions are documented in the source code.

## Organisation of faultapprox-matlab
```
├── src  : the actual python implementation of the algorithm. The "main" function is faultapprox.py.
├── tests: a number of 2D and 3D test cases, unrelated to the aforementioned paper
└── utils: utility python functions
```
It is required to run the tests from the main directory of this library.

### Python version used for development:
Python version 3.9.7