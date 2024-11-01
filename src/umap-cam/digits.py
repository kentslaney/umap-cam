import pynndescent
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
aknn = pynndescent.NNDescent(digits.data)
neighbors = aknn.neighbor_graph[0].flatten()
idx, count = np.unique(neighbors, return_counts=True)
plt.plot(idx / idx.size, np.sort(count))
plt.xlabel("Percentile")
plt.ylabel("Reverse Neighbors")
plt.title("Reverse Neighbors in MNIST for k=30")
plt.show()
