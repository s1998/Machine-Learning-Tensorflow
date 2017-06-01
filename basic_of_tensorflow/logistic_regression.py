import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib

x_data = np.array(0)
x_data[:,0] = np.random.rand(100).astype(np.float32)
x_data[:,1] = np.random.rand(100).astype(np.float32)
x_data[:,2] = np.random.rand(100).astype(np.float32)

y_data = []

for i in range(len(x_data)):
    y_data.append(np.random.randint(low=0.0, high=2.0))

for i in range(len(x_data)):
    print(i, x_data[i], y_data[i])

