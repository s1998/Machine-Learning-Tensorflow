import tensorflow as tf

scalar = tf.constant([5])
vector = tf.constant([5,6,7])
matrix = tf.constant([[1,2,3], [4,5,6], [7,8,9]])
tensor = tf.constant([[[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]], [[1,2,3], [4,5,6], [7,8,9]]])

with tf.Session() as sess:
    result = sess.run(scalar)
    print("Scalar", result)
    print("\n")
    result = sess.run(vector)
    print("Vector", result)
    print("\n")
    result = sess.run(matrix)
    print("Matrix", result)
    print("\n")
    result = sess.run(tensor)
    print("Tensor", result)
    print("\n")
