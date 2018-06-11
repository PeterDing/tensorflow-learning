import tensorflow as tf

sess = tf.Session()


def feeding():
    # {{{
    # Feeding
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y

    # Also note that the feed_dict argument can be used to overwrite **any tensor** in the graph.
    # The only difference between placeholders and other tf.Tensors is that
    # placeholders throw an error if no value is fed to them.
    print(sess.run(z, feed_dict={x: 3, y: 4.5}))
    print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))
    # }}}


if __name__ == '__main__':
    feeding()
    sess.close()
