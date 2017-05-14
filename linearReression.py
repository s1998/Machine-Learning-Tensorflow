import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 3 + 2

# there is a lot of explaining to do here :)
# lambda y: y*2
# function-: input is y and output is 2*y
# so  " lambda y: y + np.random.normal(loc=0.0, scale=0.1) " creates a temporary function (func-1) and return it
# vectorize will convert a scalar function to a vector function
# so  " np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1)) " creates another temporary function(func-2)
# and works same as func-1 except that it takes vector input
# np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data) is passed y_data as input
# returns a vector which is output
# of the temporarily created function,
y_data = np.vectorize(lambda y: y + np.random.normal(loc=0.0, scale=0.1))(y_data)

a = tf.Variable(1.0)
b = tf.Variable(1.0)

y = x_data*a + b
loss = tf.reduce_mean(tf.square(y_data-y))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

train_data = []

# with tf.Sesssion() as sess:
#     pass

for step in range(100):
    evals = sess.run([train, a, b])[1:]
    if (step%5 == 0):
        print(step, evals)
        train_data.append(evals)


#plot the graph

converter = plt.colors
cr, cg, cb = (1.0, 1.0, 0.0)

for f in train_data:
    cb += 1.0/len(train_data)
    cg -= 1.0/len(train_data)
    if cb>1.0: cb = 1.0
    if cg<0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x : x*a + b)(x_data)
    line = plt.plot(x_data, f_y)
    plt.setp(line, color=(cr, cg, cb))

plt.plot(x_data, y_data, 'ro')

green_line = mpatches.Patch(color='red', label = ' Data Points ')

plt.legend(handles = [green_line])

plt.show()
