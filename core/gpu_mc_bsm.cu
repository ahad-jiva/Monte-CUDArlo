#include <cmath>
#include <chrono>
#include <iostream>
#include <random>
#include <curand_kernel.h>

__global__ void monte_carlo_eu_call(double *payoffs, double S0, double K, double T, double r, double sigma, unsigned int seed, int n) {
    // All parameters will be the same for each thread
    // Only the random normal sample will be per-thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    curandState state;
    curand_init(seed, idx, 0, &state);
    double Z = curand_normal_double(&state);

    double S_T = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);

    payoffs[idx] = max(S_T - K, 0.0);

}

int main(){

    // set up parameters
    int n_paths = 100000000;
    double S0 = 100;
    double K = 150;
    double T = 10;
    double r = 0.05;
    double sigma = 0.1;

    // memory
    double *payoffs;
    cudaMalloc(&payoffs, n_paths * sizeof(double));

    // kernel config
    int block_size = 256;
    int num_blocks = (n_paths + block_size - 1) / block_size;

    auto start_time = std::chrono::high_resolution_clock::now();
    monte_carlo_eu_call<<<num_blocks, block_size>>>(payoffs, S0, K, T, r, sigma, time(NULL), n_paths);
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration<double>(end_time - start_time);

    std::vector<double> h_payoffs(n_paths);
    cudaMemcpy(h_payoffs.data(), payoffs, n_paths * sizeof(double), cudaMemcpyDeviceToHost);

    double payoff_sum = 0.0;
    for (int i = 0; i < n_paths; i++){
        payoff_sum += h_payoffs[i];
    }
    double option_price = (payoff_sum / n_paths) * exp(-r * T);

    cudaFree(payoffs);
    std::cout << "Computation on GPU took " << duration.count() << " seconds.\n";
    std::cout << "Option is worth: " << option_price << std::endl;

}