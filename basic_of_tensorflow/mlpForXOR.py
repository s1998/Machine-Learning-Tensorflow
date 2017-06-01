import tensorflow as tf

xData = tf.constant([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
yData = tf.constant([[0.0], [1.0], [1.0], [0.0]])

w1 = tf.Variable(tf.random_uniform((2,2)))
b1 = tf.Variable(tf.zeros(2))

w2 = tf.Variable(tf.random_uniform((2,1)))
b2 = tf.Variable(tf.zeros(1))

# yPredicted = tf.nn.sigmoid(tf.nn.sigmoid(tf.matmul(xData, w1) + b1), w2)
yPredicted = tf.nn.sigmoid(tf.matmul(tf.nn.sigmoid( tf.matmul(xData, w1) + b1), w2) + b2)

crossEntropyLoss = - tf.reduce_sum(yData * tf.log(yPredicted) + (1-yData)*tf.log(1-yPredicted))

trainer = tf.train.GradientDescentOptimizer(0.01).minimize(crossEntropyLoss)

init = tf.global_variables_initializer()

epochs = 100000

resultsList = list()

with tf.Session() as sess:
    sess.run(init)
    for i in range(epochs):
        sess.run(trainer)
        print(i, sess.run(crossEntropyLoss))
        if(i%100==0):
            resultsList.append([i, sess.run(crossEntropyLoss), sess.run(yPredicted)])

for i in range(len(resultsList)):
    print(resultsList[i])

