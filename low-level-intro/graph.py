import tensorflow as tf

sess = tf.Session()


def graph():
    # {{{
    # a, b, c are just Tensor without value
    a = tf.constant(0.1)
    b = tf.constant(0.1)
    c = a + b

    # record to tensorboard
    # active tensorboard:
    # tensorboard --logdir='./graph-log'
    writer = tf.summary.FileWriter('./graph-log')
    writer.add_graph(tf.get_default_graph())

    # writer does write only after session must be runned
    # or sess.run({'ab-key-example': (a, b), 'c-example': c})  # key can be any string
    print(sess.run(c))
    # }}}

    # {{{
    vec = tf.random_uniform(shape=(3,))
    out1 = vec + 1
    out2 = vec + 2

    # In each run, vec has different value
    print(sess.run(vec))
    print(sess.run(vec))
    # But, in one session, vec has same value
    print(sess.run((out1, out2)))
    # }}}


if __name__ == '__main__':
    graph()
    sess.close()
