#include <cmath>
#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>

void monte_carlo_eu_call(double S0, double K, double T, double r, double sigma, int n, int steps) {

  // random number setup
  std::random_device rd{};
  std::mt19937 gen{rd()};

  double payoff_sum = 0.0;

  auto start_time = std::chrono::high_resolution_clock::now();

  std::normal_distribution<double> normal(0.0, 1.0);

  double dt = T / steps;
  
  std::vector<std::vector<double>> paths(n, std::vector<double>(steps + 1, 0.0));

  // main loop
  for (int i = 0; i < n; i++) {
    paths[i][0] = S0;
    for (int j = 1; j <= steps; j++){
      double Z = normal(gen);
      paths[i][j] = paths[i][j-1] * exp((r - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
    }
  }

  // write to csv
  std::ofstream file("bsm_paths.csv");

  // add header info for lognormal calculation in graphing
  file << S0 << "," << T << "," << r << "," << sigma << "\n";

  for (int j = 0; j <= steps; j++){
    for (int i = 0; i < n; i++){
      file << paths[i][j];
      if (i != n - 1) file << ",";
    }
    file << "\n"; 
  }
  file.close();

  std::cout << "Saved " << n << " paths with " << steps << " steps to 'paths.csv'" << std::endl;

  for (int i = 0; i < n; i++){
    payoff_sum += paths[i][steps];
  }

  // averaging and discounting
  double avg_payoff_discounted = (payoff_sum / n) * exp(-r * T);
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end_time - start_time);
  
  std::cout << "Computation took " << duration.count() << " seconds.\n";
  std::cout << "Option is worth: " << avg_payoff_discounted << std::endl;
}

int main() {
    monte_carlo_eu_call(100, 150, 10, 0.05, 0.1, 5000, 200);
}