#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

colors = {
    0: '#3800a5',
    1: 'magenta',
    2: '#e8f301'
}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i, (u1, u2, u3) in enumerate(pca_data):
    ax.scatter(u1, u2, u3, c=colors.get(labels[i]), alpha=0.5)

ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
ax.set_title('PCA of Iris Dataset')

plt.show()
