import tensorflow as tf
import numpy as np

#  Just a practice file
#  Divided into sections
#  Contains :
#    -> Linear regression
#    -> Softmax regression


# step 1 -> get the data set and the variables nedeed
# to write the algo
x_data = np.random.rand(100).astype(np.float32)
y_data = 3 * x_data + 2
y_data = np.vectorize(lambda y : y + np.random.normal(loc = 0.0, scale = 0.1))(y_data)

a = tf.Variable(1.0)
b = tf.Variable(1.0)

# step 2 -> write tha algo and do the prediction
# and obtain the result
yPredicted = a * x_data + b

# step 3 -> write the cost function
loss = tf.reduce_mean(tf.square(yPredicted - y_data))

# step 4 -> define the optimizer and the trainer
optimizer = tf.train.GradientDescentOptimizer(0.1)
train = optimizer.minimize(loss)

# step 5 -> initialize all the variables
init = tf.global_variables_initializer()

# step 6 -> Now run the entire code and do
# the computations and optimizations
resultsList= list()
with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train)
        if(i%100==0):
            aRes = sess.run(a)
            bRes = sess.run(b)
            lossRes = sess.run(loss)
            resultsList.append([i, aRes, bRes, lossRes])

for i in range(len(resultsList)):
    print("Iteration number, a, b, loss are : ", resultsList[i])


# " Softmax Regression "
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X_data = tf.placeholder(dtype=tf.float32, shape=[None, 784])
Y_data = tf.placeholder(dtype=tf.float32, shape=[None, 10])

weight = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

yPredicted = tf.nn.softmax(tf.matmul(X_data, weight) + bias)
crossEntropyLoss = - tf.reduce_sum(Y_data * tf.log(yPredicted))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(crossEntropyLoss)

init = tf.global_variables_initializer()
tf.Session().run(init)

resultsList = list()
n_epochs = 1000
correct_prediction = tf.equal(tf.argmax(Y_data,1), tf.argmax(yPredicted,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

with tf.Session() as sess:
    sess.run(init)
    for i in range(n_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        print(sess.run(crossEntropyLoss, feed_dict={X_data: batch_xs, Y_data: batch_ys}))
        sess.run(train, feed_dict={X_data: batch_xs, Y_data: batch_ys})
        if(i%100 == 0):
            loss = sess.run(crossEntropyLoss, feed_dict={X_data: batch_xs, Y_data: batch_ys})
            resultsList.append([i, loss])
    print("Accuracy after 1000 iterations : ", sess.run(accuracy, feed_dict={X_data: mnist.test.images,Y_data: mnist.test.labels}))

for i in range(10):
    print("Iteration number, loss : ", resultsList[i])


# Code from
# http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/

import numpy as np
import random
from random import shuffle
import tensorflow as tf

# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

NUM_EXAMPLES = 10000

train_input = ['{0:020b}'.format(i) for i in range(2**20)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print ("test and training data loaded")


data = tf.placeholder(tf.float32, [None, 20,1]) #Number of examples, number of input, dimension of each input
target = tf.placeholder(tf.float32, [None, 21])
num_hidden = 24
cell = tf.nn.rnn_cell.LSTMCell(num_hidden,state_is_tuple=True)
val, _ = tf.nn.dynamic_rnn(cell, data, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)
weight = tf.Variable(tf.truncated_normal([num_hidden, int(target.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[target.get_shape()[1]]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(target * tf.log(tf.clip_by_value(prediction,1e-10,1.0)))
optimizer = tf.train.AdamOptimizer()
minimize = optimizer.minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(target, 1), tf.argmax(prediction, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

batch_size = 1000
no_of_batches = int(len(train_input)) / batch_size
epoch = 5000
for i in range(epoch):
    ptr = 0
    for j in range(int(no_of_batches)):
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr+=batch_size
        sess.run(minimize,{data: inp, target: out})
    print ("Epoch ",str(i))
incorrect = sess.run(error,{data: test_input, target: test_output})
print (sess.run(prediction,{data: [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]}))
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
# tf.nn.rnn_cell.MultiRNNCell
# tf.nn.seq2seq.sequence_loss_by_example()
