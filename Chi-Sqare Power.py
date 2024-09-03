import numpy as np
from scipy.stats import hypergeom, chi2
from itertools import product
from collections import Counter
import pandas as pd


class SimulationPowerCalculator:
    def __init__(self, n_trials, known_probability=0.999999, alpha=0.0000001):
        self.n_trials = n_trials
        self.known_probability = known_probability
        self.alpha = alpha

    def expected_probability(self, n_choices, n_picks, n_false):
        """
        Calculate the expected probability using the hypergeometric distribution.
        """
        M = n_choices  # Total number of choices
        N = n_false    # Total number of false choices
        n = n_picks    # Number of picks

        # Hypergeometric distribution probabilities
        expected_probs = [hypergeom.pmf(k, M, N, n) for k in range(n_picks + 1)]
        return expected_probs

    def simulate(self, n_choices, n_picks, n_false):
        """
        Run a simulation for a given combination of parameters and calculate the observed frequencies.
        """
        results = []

        # Pre-calculate probabilities for when the known probability is applied
        p_values = np.array([n_false / (n_choices - (i + 1)) for i in range(n_picks)])
        unknown_weights = np.array([0.2, 0.8])

        for _ in range(self.n_trials):
            if np.random.rand() < self.known_probability:
                choices = []
                for p in p_values:
                    weights = [1 - p, p]
                    if min(weights) < 0:
                        weights = [0, 1]
                    choices.append(np.random.choice([0, 1], p=weights))
            else:
                choices = np.random.choice([0, 1], size=n_picks, p=unknown_weights)

            true_picks = np.sum(choices)
            results.append(true_picks)

        # Calculate the simulated probabilities
        result_counts = Counter(results)
        unique = np.array(list(result_counts.keys()))
        counts = np.array(list(result_counts.values()))
        return unique, counts

    def calculate_power(self, n_choices_range, n_false_range, n_picks_range):
        """
        Run simulations for all combinations of n_choices, n_false, and n_picks,
        and calculate the power for each combination.
        """
        power_results = []

        # Iterate over all combinations of n_choices, n_false, and n_picks
        for n_choices, n_false, n_picks in product(n_choices_range, n_false_range, n_picks_range):
            if n_false > n_choices or n_picks > n_choices or n_picks > n_false:
                continue  # Skip invalid combinations

            # Run simulation
            unique, counts = self.simulate(n_choices, n_picks, n_false)

            # Calculate non-centrality parameter for chi-square test
            effect_size = self.expected_probability(n_choices, n_picks, n_false)[0]
            ncp = self.n_trials * effect_size**2

            # Calculate critical value for chi-square test
            df = len(unique) - 1
            critical_value = chi2.ppf(1 - self.alpha, df)

            # Calculate power
            power = 1 - chi2.cdf(critical_value, df, ncp)
            power_results.append((n_choices, n_false, n_picks, power))

        return power_results


# Define ranges
n_choices_range = range(5, 11)
n_false_range = range(1, 6)
n_picks_range = range(1, 4)

# Create an instance of the class
simulator = SimulationPowerCalculator(n_trials=1000, known_probability=0.99, alpha=0.05)

# Calculate power for all combinations
results = simulator.calculate_power(n_choices_range, n_false_range, n_picks_range)

df = pd.DataFrame(results, columns=['n_choices', 'n_false', 'n_picks', 'power'])

# Save the DataFrame to a CSV file
df.to_csv('results.csv', index=False)


# Print the results
for result in results:
    print(f'n_choices: {result[0]}, n_false: {result[1]}, n_picks: {result[2]}, power: {result[3]:.4f}')