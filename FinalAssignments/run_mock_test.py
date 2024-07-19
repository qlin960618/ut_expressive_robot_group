import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 3  # sample size for each group
mean_happy = 7
mean_sad = 3
std_dev = 1.5

# Generate random data
happy_scores = np.random.normal(mean_happy, std_dev, n)
sad_scores = np.random.normal(mean_sad, std_dev, n)

# Create a DataFrame
data = pd.DataFrame({
    'Group': ['Happy'] * n + ['Sad'] * n,
    'Happiness_Score': np.concatenate([happy_scores, sad_scores])
})


print(data)


from scipy import stats

# Separate the data into two groups
happy_group = data[data['Group'] == 'Happy']['Happiness_Score']
sad_group = data[data['Group'] == 'Sad']['Happiness_Score']

# Perform t-test
t_stat, p_value = stats.ttest_ind(happy_group, sad_group)

print(t_stat, p_value)
