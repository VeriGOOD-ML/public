import tensorflow.compat.v1 as tf
import numpy as np


m = 1740

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
    c = tf.multiply(y, h)
    distances = tf.subtract(1., c)
    # maximum = tf.maximum(0., distances)
    #maximum = tf.boolean_mask(distances, tf.greater(0., distances))

    # Look here for gradient of SVM objective function: http://u.cs.biu.ac.il/~jkeshet/teaching/aml2016/sgd_optimization.pdf
    maximum = tf.cast(tf.greater(distances, 0.), tf.float32)

    g = tf.multiply(maximum, x)

    g = tf.multiply(mu, g)
    w = tf.subtract(w, g, name='update')

    sess.run(tf.initialize_all_variables())
    feed_dict = {x: x_batch, y: y_batch, w: weights, b: biases}
    sess.run(w, feed_dict)
    tf.train.Saver().save(sess, 'model.ckpt')
