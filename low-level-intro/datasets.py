import tensorflow as tf

sess = tf.Session()


def datasets1():
    # {{{
    # datasets
    my_data = [
        [
            0,
            1,
        ],
        [
            2,
            3,
        ],
        [
            4,
            5,
        ],
        [
            6,
            7,
        ],
    ]
    slices = tf.data.Dataset.from_tensor_slices(my_data)
    next_item = slices.make_one_shot_iterator().get_next()
    while True:
        try:
            print(sess.run(next_item))
        except tf.errors.OutOfRangeError:
            break
    # }}}


def datasets2():
    # if the dataset depends on stateful operations you may need to
    # initialize the iterator before using it, as shown below:
    # {{{
    # datasets
    r = tf.random_normal([10, 3])

    dataset = tf.data.Dataset.from_tensor_slices(r)
    iterator = dataset.make_initializable_iterator()
    next_row = iterator.get_next()

    sess.run(iterator.initializer)
    while True:
        try:
            print(sess.run(next_row))
        except tf.errors.OutOfRangeError:
            break
    # }}}


if __name__ == '__main__':
    print('dataset1:')
    datasets1()
    print('dataset2:')
    datasets2()
    sess.close()
