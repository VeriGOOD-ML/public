import tensorflow as tf
import numpy as np
import tf2onnx


m = 17770  # num of movies
n = 480189  # num of users
k = 5  # num of features

x_1_batch = np.random.rand(k)
x_2_batch = np.random.rand(k)

r_1_batch = np.random.rand(m)
y_1_batch = np.random.rand(m)

r_2_batch = np.random.rand(n)
y_2_batch = np.random.rand(n)


weight_1 = np.random.rand(m, k)
weight_2 = np.random.rand(n, k)


with tf.Session() as sess:

    x_1 = tf.placeholder(tf.float32, shape=(k, ), name='x1')
    x_2 = tf.placeholder(tf.float32, shape=(k, ), name='x2')

    r_1 = tf.placeholder(tf.float32, shape=(m, ), name='r1')
    y_1 = tf.placeholder(tf.float32, shape=(m, ), name='y1')

    r_2 = tf.placeholder(tf.float32, shape=(n, ), name='r1')
    y_2 = tf.placeholder(tf.float32, shape=(n, ), name='y1')

    # w = tf.Variable(np.random.rand(m), name='W', dtype=tf.float32)
    # b = tf.Variable(np.random.rand(m), name='b', dtype=tf.float32)
    w_1 = tf.placeholder(tf.float32, shape=(m, k), name='W1')
    w_2 = tf.placeholder(tf.float32, shape=(n, k), name='W2')

    b = tf.placeholder(tf.float32, shape=(m, ), name='b')

    mu = tf.constant(1, dtype=tf.float32)

    _ = tf.Variable(initial_value=np.random.rand(1))


    h_1 = tf.reduce_sum(tf.multiply(tf.linalg.matvec(w_1, x_2), r_1))
    h_2 = tf.reduce_sum(tf.multiply(tf.linalg.matvec(w_2, x_1), r_2))

    d_1 = tf.subtract(h_1, y_1)
    d_2 = tf.subtract(h_2, y_2)

    g_1 = tf.linalg.matmul(tf.reshape(d_1, [m, 1]), tf.reshape(x_2, [1, k]))
    g_2 = tf.linalg.matmul(tf.reshape(d_2, [n, 1]), tf.reshape(x_1, [1, k]))

    w_1 = tf.subtract(w_1, g_1, name='w_1')
    w_2 = tf.subtract(w_2, g_2, name='w_2')


    sess.run(tf.initialize_all_variables())
    feed_dict = {x_1: x_1_batch,
                 x_2: x_2_batch,
                 r_1: r_1_batch,
                 y_1: y_1_batch,
                 r_2: r_2_batch,
                 y_2: y_2_batch,
                 w_1: weight_1,
                 w_2: weight_2}
    sess.run([w_1, w_2], feed_dict)
    tf.train.Saver().save(sess, 'model.ckpt')
