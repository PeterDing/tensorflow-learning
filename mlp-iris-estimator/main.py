import numpy as np
import tensorflow as tf
import pandas

tf.logging.set_verbosity(tf.logging.INFO)

learning_rate = 0.01


# model
def make_net(features, labels, mode):
    # features can be dict or array(numpy)
    # if dict, key name can be anything
    x = features['x']

    # tf.layers.dense(inputs, units)
    net = tf.layers.dense(x, 10, activation=tf.nn.relu)
    net = tf.layers.dense(net, 10, activation=tf.nn.relu)

    # dropout 0.1
    dp = tf.layers.dropout(net, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(dp, 3)

    predictions = {
        # key name change be anything

        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
    }
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=eval_metric_ops)


# handle data
train_fl = '../data/iris_training.csv'
test_fl = '../data/iris_test.csv'

batch_size = 32

train_csv = np.array(pandas.read_csv(train_fl))
train_data = train_csv[:, :-1]
train_labels = train_csv[:, -1].astype('int32')

test_csv = np.array(pandas.read_csv(train_fl))
test_data = test_csv[:, :-1]
test_labels = test_csv[:, -1].astype('int32')


def main(args):
    # create estimator
    estimator = tf.estimator.Estimator(model_fn=make_net, model_dir='./model')

    # Set up logging for predictions
    # Log the values in the "Softmax" tensor with label "probabilities"
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_data},
        y=train_labels,
        batch_size=batch_size,
        num_epochs=None,
        shuffle=True)

    estimator.train(input_fn=train_input_fn, steps=3000, hooks=[logging_hook])

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_data}, y=test_labels, num_epochs=1, shuffle=False)

    eval_result = estimator.evaluate(input_fn=eval_input_fn)
    print('eval_input_fn', eval_result)


if __name__ == '__main__':
    # tf.app.run(main=None, argv=None)
    tf.app.run()
