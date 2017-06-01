import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

# reference implementation
# https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
print("Loaded the data.")

# We are going to follow these steps for model creation every time :
#
#     1. Data and all parameters given while class construction.
#        Model and data computation graph is also made then itself.
#        All the nodes in the dc-graph including predicted value, loss, accuracy
#        and optimizer will be designed here itself.
#
#     2. Then there is a function that help in prediction
#
#     3. A function that trains and optimizes which will be called
#        number of times we want to iterate (i.e. no of epochs).
#
#     4. A function that gets the cross-validation score.
#


learning_rate = 0.01
n_epochs = 10000
batch_size = 100
display_step = 1

input_each_time_step = 28
time_steps = 28
classes = 10
hidden_units = 10

class rnnForMNIST:
    def __init__(self, hidden_units, time_steps, num_classes, learning_rate):
        # getting the data, declaring the variables
        self.learning_rate = learning_rate
        self.x_input = tf.placeholder("float", [None, 28, 28])
        self.y_input = tf.placeholder("float", [None, 10])
        self.weights = tf.Variable(tf.random_uniform(shape=[hidden_units*2, num_classes], maxval=1))
        self.biases  = tf.Variable(tf.random_uniform(shape=[num_classes]))
        self.x_input_unstacked = tf.unstack(self.x_input, 28, 1)

        # writing the prediction algorithm
        self.fw_rnn_cell = rnn.LSTMCell(hidden_units, forget_bias=1.0)
        self.bw_rnn_cell = rnn.LSTMCell(hidden_units, forget_bias=1.0)
        self.outputs, self.states_fw, self.states_bw = rnn.static_bidirectional_rnn(self.fw_rnn_cell,
                                                                                    self.bw_rnn_cell,
                                                                                    self.x_input_unstacked,
                                                                                    dtype = tf.float32)
        self.y_predicted = tf.matmul(self.outputs[-1], self.weights) + self.biases

        # defining the loss function
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_predicted, labels=self.y_input)

        # define optimizer and trainer
        self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.trainer = self.optimizer.minimize(self.loss)

        # creating session and initializing variables
        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

        # get accuracy
        self.get_equal = tf.equal(tf.argmax(self.y_input, 1), tf.argmax(self.y_predicted, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.get_equal, tf.float32))

    def predict(self, x, y):
        result = self.sess.run(self.y_predicted, feed_dict={self.x_input: x, self.y_input: y})
        return result

    def optimize(self, x, y):
        result = self.sess.run(self.trainer, feed_dict={self.x_input: x, self.y_input: y})

    def cross_validate(self, x, y):
        result = self.sess.run(self.accuracy, feed_dict={self.x_input:x, self.y_input:y})
        return result


model = rnnForMNIST(hidden_units=hidden_units, time_steps=time_steps, learning_rate=learning_rate, num_classes=10)
for i in range(n_epochs):
    x, y = mnist.train.next_batch(batch_size)
    x = x.reshape(batch_size, 28, 28)
    model.optimize(x=x, y=y)
    if i % 500 == 0:
        x = mnist.test.images
        y = mnist.test.labels
        x = x.reshape(-1, 28, 28)
        print(i, model.cross_validate(x=x, y=y))


"""
Results :

With learning_rate = 0.01
Iteration number, cross-validation accuracy :
0 0.1039
500 0.8559
1000 0.9136
1500 0.9258
2000 0.9334
2500 0.9422
3000 0.9371
3500 0.9395
4000 0.9367
4500 0.9473
5000 0.9393
5500 0.9511
6000 0.9444
6500 0.9528
7000 0.9502
7500 0.9486
8000 0.9497
8500 0.9524
9000 0.9445
9500 0.9436

With learning_rate = 0.001
Iteration number, cross-validation accuracy :
0 0.1247
1000 0.7214
2000 0.8495
3000 0.9029
4000 0.9183
5000 0.9382
6000 0.9407
7000 0.9438
8000 0.948
9000 0.9521
10000 0.947
11000 0.9489
12000 0.9533
13000 0.9533
14000 0.958
15000 0.9555
16000 0.9525
17000 0.9559
18000 0.9598
19000 0.9623

"""
