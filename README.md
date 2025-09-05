# Monte-CUDArlo
GPU-accelerated Black Scholes option price

Use `core/cpu_mc_bsm.cpp` for CPU simulation. Current best results on a 5800X3D come from compiling with \
`-O3 -ffast-math -march=native` flags.

Use `core/gpu_mc_bsm.cu` for GPU simulation. Requires `nvcc` CUDA C++ compiler and a valid CUDA installation.

Both programs generate .csv files that can be visualized with `viz/plots.py` to see the price paths and the terminal price lognormal distribution.

TODO:
- <del>Basic CPU sim set up</del>
- <del>Paths saved to csv and visualized</del>
- <del>GPU sim set up and path saving</del>
- Greeks calculation (in CUDA)
- Multi-asset correlation simulations
  - Cholesky decomposition for correlation matrices
  - multivariate Brownian increments and multi-asset GBM paths
- RL hedging model
  - Gym wrapper for RL environment
  - Visualize actual hedge vs optimal hedge
