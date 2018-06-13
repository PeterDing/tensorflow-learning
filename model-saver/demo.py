import tensorflow as tf

model_pre = './model/my_mod.ckpt'

v1 = tf.get_variable('v1', shape=[2])
v2 = tf.get_variable('v2', shape=[2])

o1 = v1.assign(v1 + 1)
o2 = v2.assign(v2 - 1)

init = tf.global_variables_initializer()

# save model
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)

    o1.op.run()
    o2.op.run()

    print('init')
    print('v1', v1.eval())
    print('v2', v2.eval())

    save_path = saver.save(sess, model_pre)
    print(save_path)

# restore model
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, model_pre)

    print('restore 1')
    print('v1', v1.eval())
    print('v2', v2.eval())

# restore partial model
saver = tf.train.Saver({'v1': v1})
with tf.Session() as sess:
    v2.initializer.run()

    saver.restore(sess, model_pre)

    print('restore 2')
    print('v1', v1.eval())
    print('v2', v2.eval())

# Inspect variables in a checkpoint
# import the inspect_checkpoint library
from tensorflow.python.tools import inspect_checkpoint as chkp

print('\nvariable in checkpoint:')

# print all tensors in checkpoint file
print('all variables')
chkp.print_tensors_in_checkpoint_file(model_pre, tensor_name='', all_tensors=True)

# print only tensor v1 in checkpoint file
print('only v1')
chkp.print_tensors_in_checkpoint_file(model_pre, tensor_name='v1', all_tensors=False)

# print only tensor v2 in checkpoint file
print('only v2')
chkp.print_tensors_in_checkpoint_file(model_pre, tensor_name='v2', all_tensors=False)
