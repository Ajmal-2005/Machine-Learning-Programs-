import numpy as np
from scipy.stats import norm

# Generate synthetic 1D data from two Gaussians
np.random.seed(0)
data = np.concatenate([np.random.normal(0, 1, 100), np.random.normal(5, 1, 100)])

# Initialization
k = 2  # number of Gaussians
n = len(data)
means = np.random.choice(data, k)
variances = np.full(k, np.var(data))
weights = np.full(k, 1 / k)

# EM algorithm
for iteration in range(10):
    # E-step: compute responsibilities
    responsibilities = np.zeros((n, k))
    for i in range(k):
        responsibilities[:, i] = weights[i] * norm.pdf(data, means[i], np.sqrt(variances[i]))
    responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)

    # M-step: update parameters
    for i in range(k):
        resp_sum = responsibilities[:, i].sum()
        means[i] = (responsibilities[:, i] @ data) / resp_sum
        variances[i] = ((responsibilities[:, i] * (data - means[i]) ** 2).sum()) / resp_sum
        weights[i] = resp_sum / n

    print(f"Iteration {iteration+1}: Means={means}, Variances={variances}, Weights={weights}")
