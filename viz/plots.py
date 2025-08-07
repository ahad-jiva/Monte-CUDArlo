import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import lognorm

with open("../core/bsm_paths.csv", "r") as f:
    header = f.readline()
    parameters = header.split(",")
    S0 = float(parameters[0])
    T = float(parameters[1])
    r = float(parameters[2])
    sigma = float(parameters[3])
    

df = pd.read_csv("../core/bsm_paths.csv", header=None, skiprows=1)

terminal_prices = df.iloc[-1]

mu = np.log(S0) + (r - 0.5 * sigma**2) * T
sigma_t = sigma * np.sqrt(T)
x_vals = np.linspace(min(terminal_prices), max(terminal_prices), 1000)
lognorm_pdf = lognorm.pdf(x_vals, s=sigma_t, scale=np.exp(mu))

fig, axs = plt.subplots(1, 2, figsize=(14, 5))

# paths
for col in df.columns:
    axs[0].plot(df[col], alpha=0.8)
axs[0].set_title("Simulated Asset Price Paths")
axs[0].set_xlabel("Time Step")
axs[0].set_ylabel("Asset Price")
axs[0].grid(True)

# histogram
axs[1].hist(terminal_prices, bins=50, density=True, alpha=0.6, edgecolor='black', label='Simulated')
axs[1].plot(x_vals, lognorm_pdf, 'r', linewidth=2, label='Lognormal PDF')
axs[1].set_title("Histogram of Terminal Prices")
axs[1].set_xlabel("S_T")
axs[1].set_ylabel("Probability Density")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()