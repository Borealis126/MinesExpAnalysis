# Overview
This project makes it easy to manipulate and plot the data used in the 
two-qubit paper. It must be run in a directory containing ExpData, 
which can be found 
[here](https://drive.google.com/file/d/191BVc0W6bm6WQ7XinuNCUlevVMtIJ1YW/view?usp=sharing)
(make sure to unzip it).

The user interface is analysis.ipynb. There the data for the various runs
can be loaded (via input in the User Parameters section).

Keywords used in the Analysis section:
- basePath: the file directory of the requested run in ExpData.
- date: The date the data was collected. Also the date of the corresponding 
Identity reference used for SPAM correction.
- gateIndices: Leave as range(20) unless wanting to analyze only a subset
of the fidelity curve.

In the Analysis section there are eight functions used that are loaded from
analysis_funcs.py:

- gateData: Returns a dictionary containing 
the calculated gate unitary and gate durations.
- getTomos: Returns the tomography
object for each of the gate indices. The _R_mle method of these objects 
contains the relevant PTM, and the plot() method displays the input/output
colormap.
- plotPTMMap: self-explanatory
- plotPTMMap_theory: self-explanatory
- IFs: Returns a dictionary containing the IF data for the gateIndices. 
- fidelityCurve: Plots the fidelity curves based on IFs results.
- IFs_gaussianNoise: Applies gaussian noise with variance 1 to the PTM.
Returns dictionary containing IF mean/std data.
- fidelityCurve_noise: Plots fidelity curves based on IFs_gaussianNoise 
results.

# Requirements
A python 3.10 environment. Run the following commands to install dependencies:
```
conda install numpy scipy matplotlib pandas scikit-learn jupyter notebook sympy

conda install -c conda-forge cvxpy qutip

pip install forest-benchmarking
```
And then install pytorch via the 
installation instructions [here](https://pytorch.org/)