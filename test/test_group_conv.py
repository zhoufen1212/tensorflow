import tensorflow as tf

#data order is [batch, in_height, in_width, in_channels], NHWC
d_c0 = tf.constant([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7], shape=[1,1,8,1])
d_c1 = tf.constant([2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7], shape=[1,1,8,1])
d = tf.concat(values=[d_c0, d_c1], axis=3)

#weight order is [filter_height, filter_width, in_channels, out_channels]
w_c0 = tf.constant([11.0, 11.1, 11.2], shape=[1,3,1,1])
w_c1 = tf.constant([21.0, 21.1, 21.2], shape=[1,3,1,1])
w = tf.concat(values=[w_c0, w_c1], axis=3)

o = tf.nn.conv2d(d, w, strides=[1,1,1,1], padding="SAME")

#o_err = tf.losses.mean_squared_error(o, o1)


