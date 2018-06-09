import tensorflow as tf
import tensorflow.contrib.eager as tfe

# active eager model
tfe.enable_eager_execution()

# handle data
fn = '../data/iris_training.csv'


def parse_csv(line):
    default = [[0.], [0.], [0.], [0.], [0]]
    line = tf.decode_csv(line, default)
    feature = tf.reshape(line[:-1], shape=(4,))
    label = tf.reshape(line[-1], shape=())
    return feature, label


batch_size = 32

dataset = tf.data.TextLineDataset(fn) \
    .skip(1) \
    .map(parse_csv) \
    .shuffle(buffer_size=1000) \
    .batch(batch_size)

# model
net = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3)
])


# loss
def loss(net, x, y):
    y_ = net(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(net, x, y):
    with tf.GradientTape() as tape:
        l = loss(net, x, y)
    return tape.gradient(l, net.variables)


# optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

# train
train_loss_results = []
train_accuracy_results = []

epochs = 200

for epoch in range(epochs):
    loss_avg = tfe.metrics.Mean()
    loss_acc = tfe.metrics.Accuracy()

    for data, label in dataset:
        grad_ = grad(net, data, label)
        optimizer.apply_gradients(
            zip(grad_, net.variables), global_step=tf.train.get_or_create_global_step())

        loss_avg(loss(net, data, label))
        loss_acc(tf.argmax(net(data), axis=1, output_type=tf.int32), label)

    train_loss_results.append(loss_avg.result())
    train_accuracy_results.append(loss_acc.result())

    if epoch % 50 == 0:
        print('epoch: {}, loss: {}, acc: {}'.format(epoch, loss_avg.result(),
                                                    loss_acc.result()))
