from __future__ import division
from __future__ import print_function
from model_util import Model_Configuration
import model_util as util

import tensorflow as tf


args = util.get_args("test")


config = Model_Configuration()

if args.task_id is not None:
    config.babi_id = args.task_id

config.strong_supervision = False

config.train_mode = False

print('Testing DMN on babi task', config.babi_id)

# create model
with tf.variable_scope('DMN') as scope:
    from rnn import RNN

    model = RNN(config)

print('initializing variables')
init = tf.global_variables_initializer()
model_saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)

    print('restoring model weights')
    model_saver.restore(session, 'model_weights/task' + str(model.config.babi_id) + '.weights')

    print('testing DMN')
    test_loss, test_accuracy = model.run_epoch(session, model.test)

    print('')
    print('Test accuracy:', test_accuracy)
