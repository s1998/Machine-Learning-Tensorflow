# Basic RNN implementation using optimizer
# (gradient descent) from tensorflow
# Inspired from - > https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
# http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/

import tensorflow as tf
import numpy as np

# Create a very simple binary dataset
echoStep = 5

X_data = np.zeros([50000, 1])
Y_data = np.zeros([50000, 1])
for i in range(50000 - echoStep):
    X_data[i, 0] = np.random.randint(low=0, high=2)
    Y_data[i + echoStep, 0] = X_data[i, 0]

