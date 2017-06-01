import tensorflow as tf

# Understanidn tensors and
# the reshape function of tensorflow
a = tf.constant([ [[[1], [2]], [[3], [4]]],
                  [[[5], [6]], [[7], [8]]]])
# a tensor containing
# 2 3-d tensors
# each 3d tensor has 2 two 2-d tensors of shape
# 2*1
sess = tf.Session()

print(tf.shape(a))
print(sess.run(a))
print(tf.shape(a[0]))
print(sess.run(a[0]))
print(tf.shape(a[0][0]))
print(sess.run(a[0][0]))
print(tf.shape(a[0][0][0]))
print(sess.run(a[0][0][0]))

a = tf.constant([[1,2,3,4,5,6], [4,5,6,7,8,9]])
a = tf.reshape(a, [2,2,3])
print(sess.run(a))
