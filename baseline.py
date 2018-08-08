from __future__ import print_function

import argparse
import re
from functools import reduce

import numpy as np
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from pandas import DataFrame

import model_util as util

word_idx = {}
story_maxlen = 0
query_maxlen = 0
reverse_word_map = {}


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
     tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if
            not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)


def vectorize(data, word_idx, l_maxlen):
    xs = []
    for story in data:
        print(story, type(story))
        x = word_idx.get(story, 0)
        xs.append(x)
    return pad_sequences(xs, maxlen=l_maxlen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--task_id", help="specify babi task 1-20 (default=1)")
    args = parser.parse_args()
    task_id = args.task_id
    # task_id = args
    RNN = recurrent.GRU
    EMBED_HIDDEN_SIZE = 50
    SENT_HIDDEN_SIZE = 100
    QUERY_HIDDEN_SIZE = 100
    BATCH_SIZE = 32
    EPOCHS = 40
    print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                               EMBED_HIDDEN_SIZE,
                                                               SENT_HIDDEN_SIZE,
                                                               QUERY_HIDDEN_SIZE))

    challenge = util.get_task_name(task_id)

    train_file = open("./data/en-10k/" + challenge + "_train.txt", "r")
    test_file = open("./data/en-10k/" + challenge + "_test.txt", "r")
    train = get_stories(train_file)
    test = get_stories(test_file)

    vocab = set()
    for story, q, answer in train + test:
        vocab |= set(story + q + [answer])
    vocab = sorted(vocab)
    print(vocab)

    vocab_size = len(vocab) + 1
    print("vocab size:", vocab_size)
    word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
    print("word indices dict:", word_idx)
    story_maxlen = max(map(len, (x for x, _, _ in train + test)))
    print("max len of story:", story_maxlen)
    query_maxlen = max(map(len, (x for _, x, _ in train + test)))
    print("max len of query:", query_maxlen)

    input_file_name = "./baseline/inputs/input_" + task_id + ".txt"
    input_file = open(input_file_name, "w")
    input_file.write(str(word_idx))
    input_file.write("\n")
    input_file.write(str(story_maxlen))
    input_file.write("\n")
    input_file.write(str(query_maxlen))

    x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
    tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

    print('x.shape = {}'.format(x.shape))
    print('xq.shape = {}'.format(xq.shape))
    print('y.shape = {}'.format(y.shape))

    model = compose_model(EMBED_HIDDEN_SIZE, RNN, query_maxlen, story_maxlen, vocab_size)

    print('Training')
    model.fit([x, xq], y,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_split=0.05)

    loss, acc = model.evaluate([tx, txq], ty,
                               batch_size=BATCH_SIZE)
    print('Accuracy: %f' % (acc * 100))
    model_dir, weight_dir = save_model_to_disk(model, task_id)

    loaded_model = load_model_from_disk(model_dir, weight_dir)

    # evaluate loaded model on test data
    loaded_model.compile(optimizer='adam',
                         loss='categorical_crossentropy',
                         metrics=['accuracy'])
    print(loaded_model.summary())
    score = loaded_model.evaluate([tx, txq], ty,
                                  batch_size=BATCH_SIZE)
    reverse_word_map = dict(map(reversed, word_idx.items()))

    sen_list = []
    for i, c in enumerate(tx):
        curr_sen = []
        for y in c:
            if y !=0:
                curr_sen.append(reverse_word_map[y])
        sen_list.append(" ".join(curr_sen))

    qtn_list = []
    for i, q in enumerate(txq):
        #     print("qtn num",i,end=' ')
        curr_qtn = []
        for y in q:
            #         print(reverse_word_map[y], end=' ')
            curr_qtn.append(reverse_word_map[y])
        # print()
        qtn_list.append(" ".join(curr_qtn))

    predictions = loaded_model.predict([tx, txq])
    print("predictions:", len(predictions[0]))
    print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1] * 100))
    actual_ans_list = []
    exp_ans_list = []
    correct_list = []
    for i, x in enumerate(predictions):
        #     print("Actual Answer for qtn:{0} is {1}".format(i, reverse_word_map[np.where(ty[i]==ty[i].max())[0][0]]))
        exp_ans_list.append(reverse_word_map[np.where(ty[i] == ty[i].max())[0][0]])
        #     print("Answer for qtn:{0}:".format(i),end=' ')
        max_value = np.amax(x)
        indices = np.where(x == x.max())
        #     print(reverse_word_map[indices[0][0]])
        actual_ans_list.append(reverse_word_map[indices[0][0]])
        if reverse_word_map[indices[0][0]] == reverse_word_map[np.where(ty[i] == ty[i].max())[0][0]]:
            correct_list.append("Correct")
        else:
            correct_list.append("Incorrect")

    df = DataFrame(
        {'Correct/Incorrect': correct_list, 'Expected Answer': exp_ans_list, 'Actual Answer': actual_ans_list,
         'Question': qtn_list, 'Sentence': sen_list},
        columns=['Sentence', 'Question', 'Actual Answer', 'Expected Answer', 'Correct/Incorrect'])
    test_file_name = "./baseline/output/test" + task_id + ".xlsx"
    df.to_excel(test_file_name, sheet_name='sheet1', index=False)

    print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))


def load_model_from_disk(model_dir, weight_dir):
    # load json and create model
    json_file = open(model_dir, 'r')
    # loaded_model_json = json_file.read()
    # json_file.close()
    loaded_model = model_from_json(json_file.read())
    # load weights into new model
    loaded_model.load_weights(weight_dir)
    print("Loaded model from disk")
    return loaded_model


def save_model_to_disk(model, task_id):
    # serialize model to JSON
    model_json = model.to_json()
    model_dir = "./baseline/model/model_{}.json".format(task_id)
    weight_dir = "./baseline/weight/weight_{}.h5".format(task_id)
    with open(model_dir, "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(weight_dir)
    print("Saved model to disk")
    return model_dir, weight_dir


def compose_model(EMBED_HIDDEN_SIZE, RNN, query_maxlen, story_maxlen, vocab_size):
    sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
    encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
    encoded_sentence = layers.Dropout(0.3)(encoded_sentence)
    question = layers.Input(shape=(query_maxlen,), dtype='int32')
    encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
    encoded_question = layers.Dropout(0.3)(encoded_question)
    encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
    encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)
    merged = layers.add([encoded_sentence, encoded_question])
    merged = RNN(EMBED_HIDDEN_SIZE)(merged)
    merged = layers.Dropout(0.3)(merged)
    preds = layers.Dense(vocab_size, activation='softmax')(merged)
    model = Model([sentence, question], preds)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print(model.summary())
    return model


def parse(seq):
    line = seq.strip()
    # line = seq.lower()
    if line.find('.'):
        line = line.replace('.', ' . ')
    if line.find('?'):
        line = line.replace('?', ' ? ')
    line = line.split()
    print(line)
    return line



if __name__ == '__main__': main()
