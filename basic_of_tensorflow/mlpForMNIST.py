import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

xData = tf.placeholder(tf.float32, [None, 784])
yData = tf.placeholder(tf.float32, [None, 10])

w1 = tf.Variable(tf.random_uniform((784, 10)))
b1 = tf.Variable(tf.ones(10))

w2 = tf.Variable(tf.random_uniform((10,10)))
b2 = tf.Variable(tf.ones(10))

batchSize = 100
yPredicted = tf.nn.softmax(tf.matmul(tf.nn.softmax( tf.matmul(xData, w1) + b1), w2) + b2)
crossEntropyLoss = - tf.reduce_sum(yData * tf.log(yPredicted) + (1-yData)*tf.log(1-yPredicted)) / batchSize

trainer = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropyLoss)
init = tf.global_variables_initializer()
epochs = 200000

resultsList = list()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        batch = mnist.train.next_batch(batchSize)
        sess.run(trainer, feed_dict={xData: batch[0], yData: batch[1]})
        print(i, sess.run(crossEntropyLoss, feed_dict={xData: batch[0], yData: batch[1]}))
        if(i%10000==0):
            crossEntropyLossRes = sess.run(crossEntropyLoss, feed_dict={xData: batch[0], yData: batch[1]})
            accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yData, 1), tf.argmax(yPredicted,1)), "float"))
            accuracyRes = sess.run(accuracy, feed_dict={xData: mnist.test.images, yData: mnist.test.labels})
            resultsList.append([i, crossEntropyLossRes, accuracyRes])
            print("i%100 is zero here, accuracy, crossEntropyloss",i, accuracyRes, crossEntropyLossRes)

for i in range(len(resultsList)):
    print(resultsList[i])

"""
Results :
Iteration, accuracy, crossEntropyloss  :  445300 0.9134 0.443774
[0, 3.2503119, 0.087800004]
[10000, 1.7406912, 0.57020003]
[20000, 1.6069738, 0.6128]
[30000, 1.4832816, 0.67860001]
[40000, 1.5463017, 0.71740001]
[50000, 1.2834016, 0.74760002]
[60000, 0.97458804, 0.76520002]
[70000, 1.3190548, 0.7773]
[80000, 1.2420336, 0.78960001]
[90000, 0.92108065, 0.81550002]
[100000, 0.66796571, 0.88380003]
[110000, 0.76646453, 0.8926]
[120000, 0.56944209, 0.89719999]
[130000, 0.66912919, 0.90009999]
[140000, 0.66693985, 0.9012]
[150000, 0.52102828, 0.9012]
[160000, 0.67749387, 0.90380001]
[170000, 0.47545701, 0.90450001]
[180000, 0.44286281, 0.90469998]
[190000, 0.59392542, 0.9059]

"""
