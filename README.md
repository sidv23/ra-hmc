# RAHMC

This repository contains the code accompanying: 

> Repelling-Attracting Hamiltonian Monte Carlo<br>
> _Siddharth Vishwanath and Hyungsuk Tak_<br>
> [arXiv:2403.04607](https://arxiv.org/abs/2403.04607)

## Getting Started

To get started, first clone the repository and start Julia.

```bash
$ git clone https://github.com/sidv23/ra-hmc.git
$ cd ./ra-hmc
$ julia
```

From the Julia REPL, you can enter the package manager by typing `]`, and activate the project environment with the required packages from the `Project.toml` file as follows.

```julia
julia> ]
pkg> activate .
pkg> instantiate
```

Alternatively, if you use the `DrWatson.jl` package, then you can quickly activate the project environment as follows.
```julia
julia> using DrWatson
julia > @quickactivate "ra-hmc"
```

If you use Github Codespaces, the `.devcontainer/devcontainer.json` file is configured to setup the right Julia environment for you on initialization. 

## Contents

The [notebooks](./notebooks/) directory contains the Jupyter notebooks for the experiments and simulations. The directory contains the following files:

- [x] `unimodal.ipynb`: Comparison of RAHMC and HMC mixing for an standard (unimodal and isotropic) Gaussian distributions when $d \in \{2, 5, 10, 50, 100\}$.

- [x] `benchmark.ipynb`: Comparison of RAHMC, HMC, RAM and PEHMC for the benchmark dataset from [Kou et al. (2005)](https://projecteuclid.org/journals/annals-of-statistics/volume-34/issue-4/Equi-energy-sampler-with-applications-in-statistical-inference-and-statistical/10.1214/009053606000000515.full) comprising of a mixture of 20 Gaussian distributions in 2 dimensions.

- [x] `circles-concentric.ipynb` and `circles-nested.ipynb`: Comparison of RAHMC, HMC, RAM, PEHMC and Wormhole HMC for distributions with "higher dimensional modes" supported on the boundary of $\ell_1$ balls in $\mathbb{R}^d$

- [x] `anisotropic.ipynb`: Comparison of RAHMC, HMC, RAM and PEHMC for a mixture of two Gaussian distributions with correlation structure in high dimensions.

- [x] `funnel.ipynb`: Comparison of RAHMC and HMC for a mixture of two funnel distributions ([Neal, 2003](https://projecteuclid.org/journals/annals-of-statistics/volume-31/issue-3/Slice-sampling/10.1214/aos/1056562461.full)).

- [x] `tuning.ipynb`: Illustration of Nesterov dual-averaging for tuning the parameters of RAHMC in high dimensions.

- [x] `bayesian_nn.ipynb`: Comparison of RAHMC and HMC for training a Bayesian neural network with known multimodality.

The [scripts](./scripts/) directory contains the `.jl` source-code for the notebooks.


## Troubleshooting

The code here uses `Julia v1.10`. For any issues, please click [here](https://github.com/sidv23/ra-hmc/issues/new/choose).
