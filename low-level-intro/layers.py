import tensorflow as tf

sess = tf.Session()

# A trainable model must modify the values in the graph to get new outputs with the same input.
# Layers are the preferred way to add trainable parameters to a graph.
# Layers package together both the variables and the operations that act on them.
# For example a densely-connected layer performs a weighted sum across all inputs for each output
# and applies an optional activation function.
# The connection weights and biases are managed by the layer object.


def layers():
    x = tf.placeholder(tf.float32, shape=[None, 3])
    # The layer inspects its input to determine sizes for its internal variables.
    # So here we must set the shape of the x placeholder so that the layer can
    # build a weight matrix of the correct size.
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)

    # Initializing Layers
    # Also note that this global_variables_initializer only initializes variables
    # that existed in the graph when the initializer was created.
    # So the initializer should be one of the last things added during graph construction.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Executing Layers
    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))


def layers_shortcut():
    # Layer Function shortcuts
    # For each layer class (like tf.layers.Dense) TensorFlow also supplies a shortcut function (like tf.layers.dense).
    # The only difference is that the shortcut function versions create and run the layer in a single call.
    #
    # While convenient, this approach allows no access to the tf.layers.Layer object.
    # This makes introspection and debugging more difficult, and layer reuse impossible.
    x = tf.placeholder(tf.float32, shape=[None, 3])
    y = tf.layers.dense(x, units=1)

    init = tf.global_variables_initializer()
    sess.run(init)

    print(sess.run(y, {x: [[1, 2, 3], [4, 5, 6]]}))


if __name__ == '__main__':
    layers()
    layers_shortcut()
    sess.close()
