import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n = 64  # sample size for each group
mean_happy = 0.7
mean_angry = 0
std_dev_angry = 1
std_dev_happy = 1

effect_size = abs(mean_happy - mean_angry)/std_dev_happy
print(effect_size)

# Generate random data
happy_scores = np.random.normal(mean_happy, std_dev_happy, n)
angry_scores = np.random.normal(mean_angry, std_dev_angry, n)

# Create a DataFrame
data = pd.DataFrame({
    'Group': ['Happy'] * n + ['Angry'] * n,
    'Happiness_Score': np.concatenate([happy_scores, angry_scores])
})


print(data)


from scipy import stats

# Separate the data into two groups
happy_group = data[data['Group'] == 'Happy']['Happiness_Score']
angry_group = data[data['Group'] == 'Angry']['Happiness_Score']

# Perform t-test
t_stat, p_value = stats.ttest_ind(happy_group, angry_group)

print(t_stat, p_value)


# plot distribution
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


# Plotting the distributions
plt.figure(figsize=(7, 3))

# Density plot for Happy group
sns.kdeplot(happy_group, shade=True, color="blue", label="Happy Group")

# Density plot for angry group
sns.kdeplot(angry_group, shade=True, color="red", label="Angry Group")

# Adding titles and labels
plt.title('Distribution of Happiness Scores')
plt.xlabel('Happiness Score')
plt.ylabel('Density')
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
