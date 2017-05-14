import tensorflow as tf

#source operations which don't require any input
a = tf.constant([2])
b = tf.constant([3])

#computational node present in the graph
c = tf.add(a,b)

#running the data flow grap
with tf.Session() as sess:
    result = sess.run(c)
    print(result)

