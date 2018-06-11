import tensorflow as tf

sess = tf.Session()


def train():
    # data
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    # model
    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)

    # loss
    loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.01)
    train = optimizer.minimize(loss)

    # init
    init = tf.global_variables_initializer()
    sess.run(init)

    # training
    for i in range(100):
        _, loss_value = sess.run((train, loss))
        print(loss_value)

    # predict
    print(sess.run(y_pred))


if __name__ == '__main__':
    train()
    sess.close()
