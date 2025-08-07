#include <cmath>
#include <chrono>
#include <iostream>
#include <random>

void monte_carlo_eu_call(double S0, double K, double T, double r, double sigma, int n) {

  // random number setup
  std::random_device rd{};
  std::mt19937 gen{rd()};

  double payoff_sum = 0.0;

  auto start_time = std::chrono::high_resolution_clock::now();

  std::normal_distribution<double> normal(0.0, 1.0);
  
  // main loop
  for (int i = 0; i < n; i++) {
    
    double Z = normal(gen);

    // SDE step
    double S_T = S0 * exp((r - 0.5 * sigma * sigma) * T + sigma * sqrt(T) * Z);
    payoff_sum += std::max(S_T - K, 0.0);
  }
  // averaging and discounting
  double avg_payoff_discounted = (payoff_sum / n) * exp(-r * T);
  
  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration<double>(end_time - start_time);
  
  std::cout << "Computation took " << duration.count() << " seconds.\n";
  std::cout << "Option is worth: " << avg_payoff_discounted << std::endl;
}

int main() {
    monte_carlo_eu_call(100, 150, 10, 0.05, 0.1, 100000000);
}