# tf.train.Supervisor Flow

Summarized from `tensorflow-models/tutorials/rnn/ptb/ptb_word_lm.py`

## Abstractive Flow

```python
import tensorflow as tf

model = build_model()

save_path = 'path/to/model'
soft_placement = False       # True for GPU

with tf.Graph().as_default():
    sv = tf.train.Supervisor(logdir=save_path)
    config_proto = tf.ConfigProto(allow_soft_placement=soft_placement)
    with sv.managed_session(config=config_proto) as session:
        # here, doing model training

        # save model
        sv.saver.save(session, save_path, global_step=sv.global_step)
```
