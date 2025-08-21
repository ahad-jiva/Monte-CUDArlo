#include <cmath>
#include <chrono>
#include <iostream>
#include <fstream>
#include <random>
#include <curand_kernel.h>

__global__ void monte_carlo_eu_call(double *paths, double S0, double K, double T, double r, double sigma, unsigned int seed, int n_paths, int n_steps) {
    // All parameters will be the same for each thread
    // Only the random normal sample will be per-thread
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_paths) return;

    curandState state;
    curand_init(seed, idx, 0, &state);

    double dt = T / (n_steps - 1);
    double S = S0;

    paths[idx * n_steps + 0] = S;

    for (int step = 1; step < n_steps; step++){
        double Z = curand_normal_double(&state);
        S *= exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
        paths[idx * n_steps + step] = S;
    }

}

int main(){

    // set up parameters
    int n_paths = 20000;
    int n_steps = 500;
    double S0 = 100;
    double K = 150;
    double T = 10;
    double r = 0.05;
    double sigma = 0.1;

    // memory
    double *paths;
    cudaMalloc(&paths, n_paths * n_steps * sizeof(double));

    // kernel config
    int threads_per_block = 256;
    int num_blocks = (n_paths + threads_per_block - 1) / threads_per_block;

    auto start_time = std::chrono::high_resolution_clock::now();
    monte_carlo_eu_call<<<num_blocks, threads_per_block>>>(paths, S0, K, T, r, sigma, time(NULL), n_paths, n_steps);

    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(end_time - start_time);
    std::cout << "Computation on GPU took " << duration.count() << " seconds.\n";

    // copy results to host
    double *d_paths = (double *)malloc(n_paths * n_steps * sizeof(double));
    cudaMemcpy(d_paths, paths, n_paths * n_steps * sizeof(double), cudaMemcpyDeviceToHost);
    std::cout << "Data successfully copied back to host memory" << std::endl;

    cudaFree(paths);

    // write to csv
    std::ofstream file("bsm_paths.csv");
    file << S0 << "," << T << "," << r << "," << sigma << "\n";

    for (int j = 0; j < n_steps; j++) {
        for (int i = 0; i < n_paths; i++) {
            file << d_paths[i * n_steps + j];
            if (i != n_paths - 1) file << ",";
        }
        file << "\n";
}
file.close();
    std::cout << "Paths written to bsm_paths.csv\n";

}