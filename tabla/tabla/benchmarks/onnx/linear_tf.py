import tensorflow as tf
import numpy as np
import tf2onnx


m = 54

x_batch = np.random.rand(m)
y_batch = np.random.rand(1)

weights = np.random.rand(m)
biases = np.random.rand(m)


with tf.Session() as sess:

    x = tf.placeholder(tf.float32, shape=(m, ), name='x')
    y = tf.placeholder(tf.float32, shape=(1, ), name='y')
    # w = tf.Variable(np.random.rand(m), name='W', dtype=tf.float32)
    # b = tf.Variable(np.random.rand(m), name='b', dtype=tf.float32)
    w = tf.placeholder(tf.float32, shape=(m, ), name='W')
    b = tf.placeholder(tf.float32, shape=(m, ), name='b')

    mu = tf.constant(1, dtype=tf.float32)

    _ = tf.Variable(initial_value=np.random.rand(1))


    h = tf.reduce_sum(tf.multiply(w, x))
    d = tf.subtract(h, y)
    g = tf.multiply(d, x)

    g = tf.multiply(mu, g)
    w = tf.subtract(w, g, name='update')


    sess.run(tf.initialize_all_variables())
    feed_dict = {x: x_batch, y: y_batch, w: weights, b: biases}
    sess.run(w, feed_dict)
    tf.train.Saver().save(sess, 'model.ckpt')
