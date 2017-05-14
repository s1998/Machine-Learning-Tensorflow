import tensorflow as tf

#simple counter creation program using tenserflow
counter = tf.Variable(0)

one = tf.constant(1)
newCounter = tf.add(counter, one)
updatedCounter = tf.assign(counter, newCounter)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run(counter))
    for i in range(9):
        sess.run(updatedCounter)
        print(sess.run(counter))

#using placeholders
a = tf.placeholder(tf.float32)
b=a*2

with tf.Session() as sess:
    # print(sess.run(b))
    # Error : InvalidArgumentError (see above for traceback): You must feed a value for placeholder tensor 'Placeholder' with dtype float
	# [[Node: Placeholder = Placeholder[dtype=DT_FLOAT, shape=[], _device="/job:localhost/replica:0/task:0/cpu:0"]()]]

    dict = {a : [[1,2,4,5], [7,2,3,4]]}
    print(sess.run(b, feed_dict = {a: 3.5}))
    print(sess.run(b, feed_dict = dict))
