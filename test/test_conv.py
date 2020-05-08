import tensorflow as tf

sess = tf.InteractiveSession()


##################################################################
# check for padding method
# weight is 3, stride = 2
##################################################################
d = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], shape=[1,1,8,1])
w = tf.constant([11.0, 11.1, 11.2], shape=[1,3,1,1])
o = tf.nn.conv2d(d, w, strides=[1,2,2,1], padding="SAME")
o.eval()

d1 = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 0.0], shape=[1,1,9,1])
w1 = w
o1 = tf.nn.conv2d(d1, w1, strides=[1,2,2,1], padding="VALID")
o1.eval()

#d1 = tf.constant([0.0, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], shape=[1,1,9,1])
#w1 = w
#o1 = tf.nn.conv2d(d1, w1, strides=[1,2,2,1], padding="VALID")
#o1.eval()


o_cmp = tf.reduce_all(tf.equal(o, o1))
o_cmp.eval()

o_diff = tf.subtract(o, o1)
o_diff.eval()

o_err = tf.losses.mean_squared_error(o, o1)
o_err.eval()

##################################################################
# check for padding method
# weight is 3, stride = 5
##################################################################
d = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4], shape=[1,1,15,1])
w = tf.constant([11.0, 11.1, 11.2], shape=[1,3,1,1])
o = tf.nn.conv2d(d, w, strides=[1,5,5,1], padding="SAME")
o.eval()

d1 = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4], shape=[1,1,15,1])
w1 = w
o1 = tf.nn.conv2d(d1, w1, strides=[1,5,5,1], padding="VALID")
o1.eval()

#d1 = tf.constant([1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3], shape=[1,1,13,1])
#w1 = w
#o1 = tf.nn.conv2d(d1, w1, strides=[1,5,5,1], padding="VALID")
#o1.eval()


o_cmp = tf.reduce_all(tf.equal(o, o1))
o_cmp.eval()

o_diff = tf.subtract(o, o1)
o_diff.eval()

o_err = tf.losses.mean_squared_error(o, o1)
o_err.eval()


