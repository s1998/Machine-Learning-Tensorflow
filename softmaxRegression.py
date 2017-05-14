import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
print("mnist")
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print("mnist")

x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

print("Here")
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
'''
tf.train.GradientDescentOptimizer
class tf.train.GradientDescentOptimizer

Optimizer that implements the gradient descent algorithm.
__init__(
    learning_rate,
    use_locking=False,
    name='GradientDescent'
)

Construct a new gradient descent optimizer.
Args:

    learning_rate: A Tensor or a floating point value. The learning rate to use.
    use_locking: If True use locks for update operations.
    name: Optional name prefix for the operations created when applying gradients. Defaults to "GradientDescent".

minimize(
    loss,
    global_step=None,
    var_list=None,
    gate_gradients=GATE_OP,
    aggregation_method=None,
    colocate_gradients_with_ops=False,
    name=None,
    grad_loss=None
)

Add operations to minimize loss by updating var_list.

This method simply combines calls compute_gradients() and apply_gradients(). If you want to process the gradient before applying them call compute_gradients() and apply_gradients()
explicitly instead of using this function.
Args:

    loss: A Tensor containing the value to minimize.
    global_step: Optional Variable to increment by one after the variables have been updated.
    var_list: Optional list or tuple of Variable objects to update to minimize loss. Defaults to the list of variables collected in the graph under the key GraphKeys.
    TRAINABLE_VARIABLES.
    gate_gradients: How to gate the computation of gradients. Can be GATE_NONE, GATE_OP, or GATE_GRAPH.
    aggregation_method: Specifies the method used to combine gradient terms. Valid values are defined in the class AggregationMethod.
    colocate_gradients_with_ops: If True, try colocating gradients with the corresponding op.
    name: Optional name for the returned operation.
    grad_loss: Optional. A Tensor holding the gradient computed for loss.

'''

'''
Other optimizer methods :
class tf.train.AdagradOptimizer
class tf.train.MomentumOptimizer
class tf.train.AdamOptimizer
'''

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
'''
tf.argmax
argmax(
    input,
    axis=None,
    name=None,
    dimension=None
)

Returns the index with the largest value across axes of a tensor.
Args:

    input: A Tensor. Must be one of the following types: float32, float64, int64, int32, uint8, uint16, int16, int8, complex64, complex128, qint8, quint8, qint32, half.
    axis: A Tensor. Must be one of the following types: int32, int64. int32, 0 <= axis < rank(input). Describes which axis of the input Tensor to reduce across.
    For vectors, use axis = 0.
    name: A name for the operation (optional).


tf.equal
equal(
    x,
    y,
    name=None
)

Returns the truth value of (x == y) element-wise.
'''

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
'''
tf.cast
cast(
    x,
    dtype,
    name=None
)
Casts a tensor to a new type.
The operation casts x (in case of Tensor) or x.values (in case of SparseTensor) to dtype.
For example:
# tensor `a` is [1.8, 2.2], dtype=tf.float
tf.cast(a, tf.int32) ==> [1, 2]  # dtype=tf.int32
Args:
    x: A Tensor or SparseTensor.
    dtype: The destination type.
    name: A name for the operation (optional).
Returns:
A Tensor or SparseTensor with same shape as x.


tf.reduce_mean
reduce_mean(
    input_tensor,
    axis=None,
    keep_dims=False,
    name=None,
    reduction_indices=None
)

Computes the mean of elements across dimensions of a tensor.
Reduces input_tensor along the dimensions given in axis. Unless keep_dims is true, the rank of the tensor is reduced by 1 for each entry in axis.
If keep_dims is true, the reduced dimensions are retained with length 1.
If axis has no entries, all dimensions are reduced, and a tensor with a single element is returned.
For example:

# 'x' is [[1., 1.]
#         [2., 2.]]
tf.reduce_mean(x) ==> 1.5
tf.reduce_mean(x, 0) ==> [1.5, 1.5]
tf.reduce_mean(x, 1) ==> [1.,  2.]
Args:

    input_tensor: The tensor to reduce. Should have numeric type.
    axis: The dimensions to reduce. If None (the default), reduces all dimensions.
    keep_dims: If true, retains reduced dimensions with length 1.
    name: A name for the operation (optional).
    reduction_indices: The old (deprecated) name for axis.

Returns:
The reduced tensor.
'''

# Initialize all the variables
init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#run the training step 1000 times!
for i in range(1000):
    print ("Running interation : ", i)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

print (sess.run(accuracy, feed_dict={x: mnist.test.images,y_: mnist.test.labels}))