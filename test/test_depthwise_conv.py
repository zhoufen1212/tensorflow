import tensorflow as tf

sess = tf.InteractiveSession()

##################################################################
# compare for depthwise_conv2d  and conv2d 
##################################################################
d_c0 = tf.constant(value=[[[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]],[[1],[2],[3],[4]]]],dtype=tf.float32)
d_c1 = tf.constant(value=[[[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]],[[1],[1],[1],[1]]]],dtype=tf.float32)
d = tf.concat(values=[d_c0,d_c1],axis=3)

w_0 = tf.constant(value=0, shape=[3,3,1,1],dtype=tf.float32)
w_1 = tf.constant(value=1, shape=[3,3,1,1],dtype=tf.float32)
w_2 = tf.constant(value=2, shape=[3,3,1,1],dtype=tf.float32)
w_3 = tf.constant(value=3, shape=[3,3,1,1],dtype=tf.float32)
w_01 = tf.concat(values=[w_0,w_1],axis=2)
w_23 = tf.concat(values=[w_2,w_3],axis=2)
w = tf.concat(values=[w_01,w_23],axis=3)

o = tf.nn.depthwise_conv2d(d, w, strides=[1,1,1,1], rate=[1,1], padding='SAME')
o.eval()

d1 = d_c0
w1 = tf.concat(values=[w_0,w_2],axis=3)

o1 = tf.nn.conv2d(d1, w1, strides=[1,1,1,1], padding="SAME")
o1.eval()

d2 = d_c1
w2 = tf.concat(values=[w_1,w_3],axis=3)

o2 = tf.nn.conv2d(d2, w2, strides=[1,1,1,1], padding="SAME")
o2.eval()

o_12 = tf.concat(values=[o1,o2],axis=3)



o_cmp = tf.reduce_all(tf.equal(o, o_12))
o_cmp.eval()

o_err = tf.losses.mean_squared_error(o, o_12)
o_err.eval()

o_diff = tf.subtract(o, o_12)
o_diff.eval()



