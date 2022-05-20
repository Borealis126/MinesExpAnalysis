# Quantum Data Process Module
## Installation
### For general data processing purpose
It is recommended to set up a new Anaconda environment for QDPM. To get all the packages needed, run lines below:

```python3
conda install numpy scipy matplotlib pandas scikit-learn jupyter notebook
conda install -c conda-forge cvxpy qutip
```

### For LabVIEW integration
LabVIEW's python node functionality has compatibility issues with Anaconda environment. It is recommended to use official Python distributions. Go to [official Python website](https://www.python.org/downloads/) and download the executable installer based on the bitness of your LabVIEW and Python version you specified in your LabVIEW program.

Convex optimization in QDPM requires BLAS and LAPACK to be installed. The easiest way is to get Intel MKL package from [official MKL website](https://software.intel.com/content/www/us/en/develop/tools/math-kernel-library/choose-download/windows.html). It is free to use but you have to register.

After installing MKL, next step would be getting MKL-integrated numpy from [Unofficial Windows Binaries for Python Extension Packages](https://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy). Choose the correct wheel file to download. For example, if you are running Python 3.8 on a 64-bit system, you should download 'numpy‑[version]+mkl‑cp38‑cp38‑win_amd64.whl'. Don't use pip to install numpy. It is painful to manually configure it with MKL settings. Then install remaining packages in command line by running

```shell
py -m pip install scipy matplotlib pandas scikit-learn jupyter notebook
```

