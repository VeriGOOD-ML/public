import tensorflow as tf
import numpy as np
import tf2onnx


l1 = 351
l2 = 1000
l3 = 40

batch_size = 1

x_batch = np.random.rand(batch_size, l1)
y_batch = np.random.rand(batch_size, l3)

w_1_data = np.random.rand(l1, l2)
b_1_data = np.random.rand(1, l2)

w_2_data = np.random.rand(l2, l3)
b_2_data = np.random.rand(1, l3)


def sigmaprime(x):
    return tf.multiply(tf.math.sigmoid(x), tf.subtract(tf.constant(1.0), tf.math.sigmoid(x)))


with tf.Session() as sess:

    x = tf.placeholder(tf.float32, shape=(batch_size, l1), name='x')
    y = tf.placeholder(tf.float32, shape=(batch_size, l3), name='y')
    # w = tf.Variable(np.random.rand(m), name='W', dtype=tf.float32)
    # b = tf.Variable(np.random.rand(m), name='b', dtype=tf.float32)
    w_1 = tf.placeholder(tf.float32, shape=(l1, l2), name='W1')
    b_1 = tf.placeholder(tf.float32, shape=(1, l2), name='b1')

    w_2 = tf.placeholder(tf.float32, shape=(l2, l3), name='W2')
    b_2 = tf.placeholder(tf.float32, shape=(1, l3), name='b2')

    mu = tf.constant(1, dtype=tf.float32)

    _ = tf.Variable(initial_value=np.random.rand(1))


    z_1 = tf.add(tf.matmul(x, w_1), b_1)
    a_1 = tf.math.sigmoid(z_1)
    z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
    a_2 = tf.math.sigmoid(z_2)

    # Forward propagate
    # hidden = tf.layers.dense(inputs=x, units=l2, activation=tf.nn.sigmoid)
    # output = tf.layers.dense(inputs=hidden, units=l3)

    # Loss function
    cost = tf.subtract(a_2, y)

    # Gradient calculation (backpropagation)
    d_z_2 = tf.multiply(cost, sigmaprime(z_2))
    d_b_2 = d_z_2
    d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

    d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
    d_z_1 = tf.multiply(d_a_1, sigmaprime(z_1))
    d_b_1 = d_z_1
    d_w_1 = tf.matmul(tf.transpose(x), d_z_1)

    #update = tf.train.AdamOptimizer(mu).minimize(cost, name='update')
    w_1 = tf.subtract(w_1, tf.multiply(mu, d_w_1), name='w_1')
    w_2 = tf.subtract(w_2, tf.multiply(mu, d_w_2), name='w_2')

    sess.run(tf.initialize_all_variables())
    feed_dict = {x: x_batch, y: y_batch, w_1: w_1_data, b_1: b_1_data, w_2: w_2_data, b_2: b_2_data}
    sess.run([w_1, w_2], feed_dict)
    tf.train.Saver().save(sess, 'model.ckpt')
