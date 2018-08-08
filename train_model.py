from __future__ import division
from __future__ import print_function

import os
import sys
import time
from model_util import Model_Configuration
import model_util as util
import tensorflow as tf

args = util.get_args("train")

config = Model_Configuration()

if args.task_id is not None:
    config.babi_id = args.task_id

config.babi_id = args.task_id if args.task_id is not None else str(1)

num_runs = 1
sys.stdout = open('./dmn_logs/file' + str(config.babi_id), 'w')
print('Training DMN on babi task', config.babi_id)

best_overall_validation_loss = float('inf')

# create model
with tf.variable_scope('DMN') as scope:
    from rnn import RNN

    model = RNN(config)

for run in range(num_runs):

    print('initializing variables for the model run num:{}'.format(run))
    init = tf.global_variables_initializer()
    model_saver = tf.train.Saver()

    with tf.Session() as session:

        train_model_dir = 'trained_models/model/' + time.strftime("%Y-%m-%d %H %M")
        if not os.path.exists(train_model_dir):
            os.makedirs(train_model_dir)
        model_writer = tf.summary.FileWriter(train_model_dir, session.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        session.run(init)

        best_validation_loss = float('inf')
        best_validation_accuracy = 0.0
        best_validation_epoch = 0

        if args.restore:
            print("restore the saved weights")
            model_saver.restore(session, 'model_weights/task' + str(model.config.babi_id) + '.weights')

        print('train the model')
        for epoch in range(config.max_epochs):
            print('Epoch {}'.format(epoch))
            start = time.time()

            train_loss, train_accuracy = model.run_epoch(
                session, model.train, epoch, model_writer,
                train_op=model.train_step, train=True, run_metadata=run_metadata)
            validation_loss, validation_accuracy = model.run_epoch(session, model.valid)
            # If training loss is lower than validation loss it means that network is overfitting.
            # To overcome it we decrease either network size or increase dropout.
            # If training loss and validation loss are equal then model is underfitting.
            # Solution is to either increase size of network by adding layers and increases num of nodes per layer.
            print('Training loss: {}'.format(train_loss))
            print('Validation loss: {}'.format(validation_loss))
            print('Training accuracy: {}'.format(train_accuracy))
            print('Vaildation accuracy: {}'.format(validation_accuracy))

            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                best_validation_epoch = epoch
                if best_validation_loss < best_overall_validation_loss:
                    print('saving the best weights attained')
                    best_overall_validation_loss = best_validation_loss
                    best_validation_accuracy = validation_accuracy
                    model_saver.save(session, 'model_weights/task' + str(model.config.babi_id) + '.weights')

            if epoch - best_validation_epoch > config.terminate_epoch_rate:
                break
            print('Time taken by epoch {0} is: {1}'.format(epoch, time.time() - start))

        print('Best validation accuracy:', best_validation_accuracy)
