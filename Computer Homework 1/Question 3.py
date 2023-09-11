# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas


# Q3 Part 5

# This function conduct the PCA algorithm according to part 3 of question.
def pca(x, l):
    x_cov = x @ x.T
    u, s, vh = np.linalg.svd(x_cov)
    return u[:, 0:l].T @ x


# This function create the x_tilda matrix in question.
def zero_center(x):
    x_bar = np.zeros(x.shape)
    sum_col = np.zeros(x.shape[0])
    for i in range(x.shape[1]):
        sum_col = sum_col + x[:, i]
    x_avg = 1 / x.shape[1] * sum_col
    for i in range(x.shape[1]):
        x_bar[:, i] = x_avg
    return x - x_bar


# Read from Excel file and convert to numpy array:
x = pandas.read_csv('iris.csv').to_numpy()
# Remove extra columns:
x = x[:, 1:5].T
# Convert type of components to float number:
x = np.array(x, dtype=float)
y = pca(zero_center(x), 2)
# Plot the 2D array in diagram with different color for each flower.
plt.plot(y[0, 0:50], y[1, 0:50], 'o', color='red')
plt.plot(y[0, 50:100], y[1, 50:100], 'o', color='green')
plt.plot(y[0, 100:150], y[1, 100:150], 'o', color='black')
plt.show()
