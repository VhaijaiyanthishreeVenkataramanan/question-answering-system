

from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf

import input_processor
from attention import Attention


class RNN(object):
    def load_data(self, debug=False):
        if self.config.train_mode:
            self.train, self.valid, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size, self.word_vocab = input_processor.load_input_data(
                self.config, split_sentences=True)
        else:
            self.test, self.word_embedding, self.max_q_len, self.max_sentences, self.max_sen_len, self.vocab_size, self.word_vocab = input_processor.load_input_data(
                self.config, split_sentences=True)
        self.encoding = _position_encoding(self.max_sen_len, self.config.embed_size)

    def add_tf_placeholders(self):
        """add data placeholder to tf graph"""
        self.question_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.max_q_len), name='question')
        self.input_placeholder = tf.placeholder(tf.int32,
                                                shape=(self.config.batch_size, self.max_sentences, self.max_sen_len), name='input')

        self.question_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,), name='qtn_len')
        self.input_len_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size,),name='input_len')

        self.answer_placeholder = tf.placeholder(tf.int64, shape=(self.config.batch_size,), name='answer')

        self.dropout_placeholder = tf.placeholder(tf.float32,name='dropout')

    def get_output_predictions(self, output):
        preds = tf.nn.softmax(output)
        pred = tf.argmax(preds, 1)
        return pred

    def add_loss_op(self, output):
        """Calculate loss"""
        loss = tf.reduce_sum(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.answer_placeholder))



        tf.summary.scalar('loss', loss)

        return loss

    def add_training_optimizations(self, loss):
        """Calculate and apply gradients"""
        opt = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate)
        gvs = opt.compute_gradients(loss)


        train_op = opt.apply_gradients(gvs)
        return train_op

    def get_question_vectors(self):
        """Get question vectors via embedding and GRU"""

        questions = tf.nn.embedding_lookup(self.embeddings, self.question_placeholder)

        gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        _, q_vec = tf.nn.dynamic_rnn(gru_cell,
                                     questions,
                                     dtype=np.float32,
                                     sequence_length=self.question_len_placeholder
                                     )
        return q_vec

    def get_input_fact_vecs(self):
        """Get fact (sentence) vectors via embedding, positional encoding and bi-directional GRU"""
        # get word vectors from embedding
        inputs = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
        # use encoding to get sentence representation
        inputs = tf.reduce_sum(inputs * self.encoding, 2)
        forward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        backward_gru_cell = tf.contrib.rnn.GRUCell(self.config.hidden_size)
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(
            forward_gru_cell,
            backward_gru_cell,
            inputs,
            dtype=np.float32,
            sequence_length=self.input_len_placeholder
        )
        # sum forward and backward output vectors
        fact_vecs = tf.reduce_sum(tf.stack(outputs), axis=0)
        # apply dropout
        fact_vecs = tf.nn.dropout(fact_vecs, self.dropout_placeholder)
        return fact_vecs

    def get_scalar_attention(self, q_vec, prev_memory, fact_vec, reuse):
        """Use question vector and previous memory to create scalar attention for current fact"""
        with tf.variable_scope("attention", reuse=reuse):
            features = [fact_vec * q_vec,
                        fact_vec * prev_memory,
                        tf.abs(fact_vec - q_vec),
                        tf.abs(fact_vec - prev_memory)]

            feature_vec = tf.concat(features, 1)
            attention = tf.contrib.layers.fully_connected(feature_vec,
                                                          self.config.embed_size,
                                                          activation_fn=tf.nn.tanh,
                                                          reuse=reuse, scope="fc1")

            attention = tf.contrib.layers.fully_connected(attention,
                                                          1,
                                                          activation_fn=None,
                                                          reuse=reuse, scope="fc2")
        return attention

    def build_episode(self, memory, q_vec, fact_vecs, hop_index):
        """Generate episode by applying attention to current fact vectors through a attention"""
        # A list of all tensors that are the current or past memory state of the attention mechanism.
        attentions = [tf.squeeze(
            self.get_scalar_attention(q_vec, memory, fv, bool(hop_index) or bool(i)), axis=1)
            for i, fv in enumerate(tf.unstack(fact_vecs, axis=1))]
        # inverse attention mask for what is retained in the state
        attentions = tf.transpose(tf.stack(attentions))
        self.attentions.append(attentions)
        attentions = tf.nn.softmax(attentions)
        attentions = tf.expand_dims(attentions, axis=-1)
        reuse = True if hop_index > 0 else False

        # concatenate fact vectors and attentions for input into attGRU
        gru_inputs = tf.concat([fact_vecs, attentions], 2)
        # reuse variables so that gru pass uses the same variable every pass
        with tf.variable_scope('attention_gru', reuse=reuse):
            _, episode = tf.nn.dynamic_rnn(Attention(self.config.hidden_size),
                                           gru_inputs,
                                           dtype=np.float32,
                                           sequence_length=self.input_len_placeholder
                                           )

        return episode

    def add_answer_module(self, rnn_output, q_vec):

        rnn_output = tf.nn.dropout(rnn_output, self.dropout_placeholder)

        output = tf.layers.dense(tf.concat([rnn_output, q_vec], 1),
                                 self.vocab_size,
                                 activation=None)

        return output

    def inference(self):

        with tf.variable_scope("question", initializer=tf.contrib.layers.xavier_initializer()):
            print('get question vectors')
            q_vec = self.get_question_vectors()

        with tf.variable_scope("input", initializer=tf.contrib.layers.xavier_initializer()):
            print('get input fact vectors')
            fact_vecs = self.get_input_fact_vecs()

        # keep track of attentions for possible strong supervision
        self.attentions = []

        # memory module
        with tf.variable_scope("memory", initializer=tf.contrib.layers.xavier_initializer()):
            print('build episodes')

            prev_memory = q_vec

            for i in range(self.config.hops):
                print('building episode', i)
                episode = self.build_episode(prev_memory, q_vec, fact_vecs, i)
                with tf.variable_scope("hop_%d" % i):
                    prev_memory = tf.layers.dense(tf.concat([prev_memory, episode, q_vec], 1),
                                                  self.config.hidden_size,
                                                  activation=tf.nn.relu)

            output = prev_memory

        with tf.variable_scope("answer", initializer=tf.contrib.layers.xavier_initializer()):
            output = self.add_answer_module(output, q_vec)
        return output

    def run_epoch(self, session, data, num_epoch=0, train_writer=None, train_op=None, verbose=2, train=False, run_metadata=None):
        config = self.config
        dropout = config.dropout
        if train_op is None:
            train_op = tf.no_op()
            dropout = 1
        total_steps = len(data[0]) // config.batch_size
        total_loss = []
        accuracy = 0
        # print("len(data[0])",len(data[0]))
        p = np.random.permutation(len(data[0]))
        qtn_plc_holdr, inp_plc_holdr, qtn_ln, ip_ln, ip_msk, ans = data
        qtn_plc_holdr, inp_plc_holdr, qtn_ln, ip_ln, ip_msk, ans = qtn_plc_holdr[p], inp_plc_holdr[p], qtn_ln[p], ip_ln[
            p], ip_msk[p], ans[p]
        # print("p:",p,len(p))
        # print("qtn_plc:",qtn_plc_holdr)
        # print("ip plc:",inp_plc_holdr)
        # print("qtn ln:",qtn_ln)
        # print("ip ln:",ip_ln)
        # print("ip msk:",ip_msk)
        # print("ans:",ans)

        for step in range(total_steps):
            print("step:",step)
            print("range start:",step * config.batch_size)
            print("range end:",(step+1)*config.batch_size)
            index = range(step * config.batch_size, (step + 1) * config.batch_size)
            feed = {self.question_placeholder: qtn_plc_holdr[index],
                    self.input_placeholder: inp_plc_holdr[index],
                    self.question_len_placeholder: qtn_ln[index],
                    self.input_len_placeholder: ip_ln[index],
                    self.answer_placeholder: ans[index],
                    self.dropout_placeholder: dropout}
            loss, pred, summary,_ = session.run(
                [self.calculated_loss, self.prediction, self.merged, train_op], feed_dict=feed)



            if train_writer is not None:
                train_writer.add_run_metadata(run_metadata, "step:%d"%(num_epoch * total_steps + step))
                train_writer.add_summary(summary, num_epoch * total_steps + step)

            answers = ans[step * config.batch_size:(step + 1) * config.batch_size]
            # print("summary:",summary)
            # print("answers:",answers)
            # print("pred:",pred)
            accuracy += np.sum(pred == answers) / float(len(answers))
            total_loss.append(loss)
            count=0
            for dk in range(step * config.batch_size, ((step + 1) * config.batch_size)-1):
                # print("qtn_plc_holder:",qtn_plc_holdr[dk])
                # print("input:",inp_plc_holdr[dk])
                print("step:",dk)
                print("word input:",self.get_input(inp_plc_holdr[dk]))
                print("qtn word:",self.get_qtn(qtn_plc_holdr[dk]))
                # print("answer:",answers[count])
                print("expected answer:",self.word_vocab[answers[count]])
                # print("pred:",pred[count])
                print("predicted answer:",self.word_vocab[pred[count]])
                count+=1
            if verbose and step % verbose == 0:
                print('{} / {} : loss = {}'.format(
                    step, total_steps, np.mean(total_loss)))
                sys.stdout.flush()

        if verbose:
            print()

        return np.mean(total_loss), accuracy / float(total_steps)

    def get_input(self, input_holder):
        new = input_holder
        for arr in new:
            for a in arr:
                print(self.word_vocab[a], end=' ')
        print()

    def get_qtn(self, qtn_holder):
        for a in qtn_holder:
            print(self.word_vocab[a], end=' ')
        print("?")

    def __init__(self, config):
        self.config = config
        self.variables_to_save = {}
        self.load_data(debug=False)
        self.add_tf_placeholders()

        # set up embedding
        self.embeddings = tf.Variable(self.word_embedding.astype(np.float32), name="Embedding")

        self.output = self.inference()
        # get output predictions
        self.prediction = self.get_output_predictions(self.output)

        self.calculated_loss = self.add_loss_op(self.output)
        self.train_step = self.add_training_optimizations(self.calculated_loss)
        self.merged = tf.summary.merge_all()





def _position_encoding(sentence_size, embedding_size):
    """Use RNN to parse sentence, tends to over-fit.
    The alternative would be to take sum of embedding
    """
    encoding = np.ones((embedding_size, sentence_size), dtype=np.float32)
    sen_len = sentence_size + 1
    emb_len = embedding_size + 1
    for i in range(1, emb_len):
        for j in range(1, sen_len):
            encoding[i - 1, j - 1] = (i - (emb_len - 1) / 2) * (j - (sen_len - 1) / 2)
    encoding = 1 + 4 * encoding / embedding_size / sentence_size
    return np.transpose(encoding)

