import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

xData = tf.placeholder(tf.float32, [None, 784])
yData = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_uniform((784, 10)))
b1 = tf.Variable(tf.zeros(10))

w2 = tf.Variable(tf.random_uniform((10,10)))
b2 = tf.Variable(tf.zeros(10))

batchSize = 500
yPredicted = tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid( tf.matmul(xData, w1) + b1), w2) + b2)
crossEntropyLoss = - tf.reduce_sum(yData * tf.log(yPredicted) + (1-yData)*tf.log(1-yPredicted)) / batchSize

trainer = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropyLoss)
init = tf.global_variables_initializer()
epochs = 10000

resultsList = list()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        batch = mnist.train.next_batch(batchSize)
        sess.run(trainer, feed_dict={xData: batch[0], yData: batch[1]})
        print(i, sess.run(crossEntropyLoss, feed_dict={xData: batch[0], yData: batch[1]}))
        if(i%100==0):
            crossEntropyLossRes = sess.run(crossEntropyLoss, feed_dict={xData: batch[0], yData: batch[1]})
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yData, 1), tf.argmax(yPredicted,1)), "float"))
            accuracyRes = sess.run(accuracy, feed_dict={xData: batch[0], yData: batch[1]})
            resultsList.append([i, crossEntropyLossRes, accuracyRes])
            print(i, accuracyRes)

for i in range(len(resultsList)):
    print(resultsList[i])

