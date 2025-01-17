# BanditPAM: Almost Linear-Time *k*-Medoids Clustering

This repo contains a high-performance implementation of BanditPAM from https://arxiv.org/abs/2006.06856. The code can be called directly from Python or C++.

If you use this software, please cite:

Mo Tiwari, Martin Jinye Zhang, James Mayclin, Sebastian Thrun, Chris Piech, Ilan Shomorony. "Bandit-PAM: Almost Linear Time *k*-medoids Clustering via Multi-Armed Bandits" Advances in Neural Information Processing Systems (NeurIPS) 2020.


```python
@inproceedings{BanditPAM,
  title={Bandit-PAM: Almost Linear Time k-medoids Clustering via Multi-Armed Bandits},
  author={Tiwari, Mo and Zhang, Martin J and Mayclin, James and Thrun, Sebastian and Piech, Chris and Shomorony, Ilan},
  booktitle={Advances in Neural Information Processing Systems},
  pages={368--374},
  year={2020}
}
```
# Requirements
This repo has the following requirements:

```
Python >= 3.5
gcc >= 5.4
```
It is highly encouraged that to use BanditPAM in a *nix machine for high performance and ease of installation. Windows is generally supported, but can be unstable depending on the C++ compiler/environment.

# Python Quickstart

## Install the repo and its dependencies:
This can be done either through PyPI (recommended)
```python
/BanditPAM/: pip install -r requirements.txt
/BanditPAM/: pip install BanditPAM
```
or through the source code via
```python
/BanditPAM/: git submodule update --init --recursive
/BanditPAM/: cd headers/carma
/BanditPAM/: mkdir build && cd build && cmake .. && make && sudo make install
/BanditPAM/: cd ../../..
/BanditPAM/: pip install -r requirements.txt
/BanditPAM/: sudo pip install .
```

### Example 1: Synthetic data from a Gaussian Mixture Model


```python
from BanditPAM import KMedoids
import numpy as np
import matplotlib.pyplot as plt

# Generate data from a Gaussian Mixture Model with the given means:
np.random.seed(0)
n_per_cluster = 40
means = np.array([[0,0], [-5,5], [5,5]])
X = np.vstack([np.random.randn(n_per_cluster, 2) + mu for mu in means])

# Fit the data with BanditPAM:
kmed = KMedoids(n_medoids = 3, algorithm = "BanditPAM")
# Writes results to gmm_log
kmed.fit(X, 'L2', "gmm_log")

# Visualize the data and the medoids:
for p_idx, point in enumerate(X):
    if p_idx in map(int, kmed.medoids):
        plt.scatter(X[p_idx, 0], X[p_idx, 1], color='red', s = 40)
    else:
        plt.scatter(X[p_idx, 0], X[p_idx, 1], color='blue', s = 10)


plt.show()
```


![png](README_files/README_7_0.png)


### Example 2: MNIST and its medoids visualized via t-SNE


```python
# Start in the repository root directory, i.e. '/BanditPAM/'.
from BanditPAM import KMedoids
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load the 1000-point subset of MNIST and calculate its t-SNE embeddings for visualization:
X = pd.read_csv('data/MNIST-1k.csv', sep=' ', header=None).to_numpy()
X_tsne = TSNE(n_components = 2).fit_transform(X)

# Fit the data with BanditPAM:
kmed = KMedoids(n_medoids = 10, algorithm = "BanditPAM")
kmed.fit(X, 'L2', "mnist_log")

# Visualize the data and the medoids via t-SNE:
for p_idx, point in enumerate(X):
    if p_idx in map(int, kmed.medoids):
        plt.scatter(X_tsne[p_idx, 0], X_tsne[p_idx, 1], color='red', s = 40)
    else:
        plt.scatter(X_tsne[p_idx, 0], X_tsne[p_idx, 1], color='blue', s = 5)


plt.show()
```
The corresponding logfile for this run, `mnist_log`, will contain the run's results
and additional statistics in a format that can be easily read into json.

## Building the C++ executable from source

Please note that it is *NOT* necessary to build the C++ executable from source to use the Python code above. However, if you would like to use the C++ executable directly, follow the instructions below.

### Option 1: Building with Docker

We highly recommend building using Docker. One can download and install Docker by following instructions at the [Docker install page](https://docs.docker.com/get-docker/). Once you have Docker installed and the Docker Daemon is running, run the following commands:

```
/BanditPAM$ chmod +x env_setup.sh
/BanditPAM$ ./env_setup.sh
/BanditPAM$ ./run_docker.sh
```

which will start a Docker instance with the necessary dependencies. Then:

```
/BanditPAM$ mkdir build && cd build
/BanditPAM/build$ cmake .. && make
```

This will create an executable named `BanditPAM` in `BanditPAM/build/src`.

### Option 2: Installing Requirements and Building Directly
Building this repository requires four external requirements:
* Cmake >= 3.17, https://cmake.org/download/
* Armadillo >= 10.5.3, http://arma.sourceforge.net/download.html
* OpenMP >= 2.5, https://www.openmp.org/resources/openmp-compilers-tools/ (OpenMP is supported by default on most Linux platforms, and can be downloaded through homebrew on MacOS. For instructions on installing homebrew, see https://brew.sh/.)
* CARMA >= 0.3.0, https://github.com/RUrlus/carma (use the `unstable` branch)

If installing these requirements from source, one can generally use the following procedure to install each requirement from the library's root folder (with CARMA used as an example here):
```
/BanditPAM$ cd headers/carma
/BanditPAM/headers/carma$ mkdir build && cd build
/BanditPAM/headers/carma/build$ cmake .. && make && make sudo install
```

Further installation information for [MacOS](docs/install_mac.md), [Linux](docs/install_linux.md), and [Windows](docs/install_windows.md) is available in the [docs folder](docs). Ensure all the requirements above are installed and then run:

```
/BanditPAM$ mkdir build && cd build
/BanditPAM/build$ cmake .. && make
```

This will create an executable named `BanditPAM` in `BanditPAM/build/src`.

## C++ Usage

Once the executable has been built, it can be invoked with:
```
/BanditPAM/build/src/BanditPAM -f [path/to/input.csv] -k [number of clusters] -v [verbosity level]
```

* `-f` is mandatory and specifies the path to the dataset
* `-k` is mandatory and specifies the number of clusters with which to fit the data
* `-v` is optional and specifies the verbosity level.

For example, if you ran `./env_setup.sh` and downloaded the MNIST dataset, you could run:

```
/BanditPAM/build/src/BanditPAM -f ../data/MNIST-1k.csv -k 10 -v 1
```

The expected output in the command line will be:
```
Medoids: 694,168,306,714,324,959,527,251,800,737
```
A file called `KMedoidsLogfile` with detailed logs during the process will also
be present.

## Testing

To run the full suite of tests, run in the root directory:

```
/BanditPAM$ python -m unittest discover -s tests
```

Alternatively, to run a "smaller" set of tests, from the main repo folder run `python tests/test_commit.py` or `python tests/test_push.py` to run a set of longer, more intensive tests.

## Reproducing Figures from the Paper

Note that some figures in the original paper were generated using the Python code at https://github.com/motiwari/BanditPAM-python. That code is not pretty, nor is it maintained. It only exists for reference and for reproducibility of the plots.

## Documentation

Documentation for BanditPAM can be found here: 
* Doxygen docs: `BanditPAM/docs/html/index.html`
* Installation information for MacOS: https://github.com/ThrunGroup/BanditPAM/blob/main/docs/install_mac.md
* Installation information for Linux: https://github.com/ThrunGroup/BanditPAM/blob/main/docs/install_linux.md
* Installation information for Windows: https://github.com/ThrunGroup/BanditPAM/blob/main/docs/install_windows.md

## Credits

Mo Tiwari wrote the original Python implementation of BanditPAM and now maintains the C++ implementation.

James Mayclin developed the initial C++ implementation of BanditPAM.


