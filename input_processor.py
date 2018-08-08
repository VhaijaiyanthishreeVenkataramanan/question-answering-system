from __future__ import division
from __future__ import print_function

import os as os
import model_util as util
from model_util import Constants as const
import numpy as np


def init_data(file_name):
    print("Loading data from %s" % file_name)
    tasks = []
    task = None
    for _, line in enumerate(open(file_name)):
        id = int(line[0:line.find(' ')])
        if id == 1:
            task = {const.Context: "", const.Question: "", const.Answer: "", const.Supporter: ""}
            count = 0
            id_map = {}

        line = line.strip()
        line = line.replace('.', ' . ')
        line = line[line.find(' ') + 1:]
        # if not a question
        if line.find('?') == -1:
            task[const.Context] += line
            id_map[id] = count
            count += 1

        else:
            qtn_index = line.find('?')
            tmp = line[qtn_index + 1:].split('\t')
            task[const.Question] = line[:qtn_index]
            task[const.Answer] = tmp[1].strip()
            task[const.Supporter] = []
            for num in tmp[2].split():
                task[const.Supporter].append(id_map[int(num.strip())])
            tasks.append(task.copy())

    return tasks


def get_babi_files(id, test_id):
    if test_id == "":
        test_id = id
    train_file_name = util.get_task_name(id)
    test_file_name = util.get_task_name(test_id)
    train_file = init_data(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_train.txt' % train_file_name))
    test_file = init_data(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data/en-10k/%s_test.txt' % test_file_name))
    return train_file, test_file


def load_glove(dim):
    glove_map = {}

    with open(("./data/glove.6B/glove.6B." + str(dim) + "d.txt")) as f:
        for line in f:
            l = line.split()
            glove_map[l[0]] = map(float, l[1:])

    return glove_map


def create_vector(word, word2vec, word_vector_size, silent=True):
    # if the word is missing from Glove, create some fake vector and store in glove!
    vector = np.random.uniform(0.0, 1.0, (word_vector_size,))
    word2vec[word] = vector
    if (not silent):
        print("%s is missing" % word)
    return vector


def process_word(word, word2vec, vocab, ivocab, word_vector_size, to_return="word2vec", silent=True):
    if not word in word2vec:
        create_vector(word, word2vec, word_vector_size, silent)
    if not word in vocab:
        next_index = len(vocab)
        vocab[word] = next_index
        ivocab[next_index] = word

    if to_return == "word2vec":
        return word2vec[word]
    elif to_return == "index":
        return vocab[word]


def process_input(data_raw, floatX, word2vec, vocab, ivocab, embed_size):
    questions = []
    inputs = []
    answers = []
    input_masks = []
    for x in data_raw:
        inp = x[const.Context].lower().split(' . ')
        inp = [w for w in inp if len(w) > 0]
        inp = [i.split() for i in inp]

        q = x[const.Question].lower().split(' ')
        q = [w for w in q if len(w) > 0]

        inp_vector = [[process_word(word=w,
                                    word2vec=word2vec,
                                    vocab=vocab,
                                    ivocab=ivocab,
                                    word_vector_size=embed_size,
                                    to_return="index") for w in s] for s in inp]

        q_vector = [process_word(word=w,
                                 word2vec=word2vec,
                                 vocab=vocab,
                                 ivocab=ivocab,
                                 word_vector_size=embed_size,
                                 to_return="index") for w in q]

        import wordcloud
        import matplotlib.pyplot as plt
        cloud = wordcloud.WordCloud(background_color='gray', max_font_size=60,
                                    relative_scaling=1).generate(' '.join(vocab.keys()))

        fig = plt.figure(figsize=(20, 10))
        plt.axis('off')
        plt.imshow(cloud)

        inputs.append(inp_vector)
        questions.append(np.vstack(q_vector).astype(floatX))
        answers.append(process_word(word=x[const.Answer],
                                    word2vec=word2vec,
                                    vocab=vocab,
                                    ivocab=ivocab,
                                    word_vector_size=embed_size,
                                    to_return="index"))

    return inputs, questions, answers, input_masks


def get_input_vectors(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    for i, t in enumerate(inputs):
        lens[i] = t.shape[0]
    return lens


def get_sentence_vectors(inputs):
    lens = np.zeros((len(inputs)), dtype=int)
    sen_lens = []
    max_sen_lens = []
    for i, t in enumerate(inputs):
        sentence_lens = np.zeros((len(t)), dtype=int)
        for j, s in enumerate(t):
            sentence_lens[j] = len(s)
        lens[i] = len(t)
        sen_lens.append(sentence_lens)
        max_sen_lens.append(np.max(sentence_lens))
    return lens, sen_lens, max(max_sen_lens)


def pad_inputs(inputs, lens, max_len, sen_lens=None, max_sen_len=None):
    padded = np.zeros((len(inputs), max_len, max_sen_len))
    for i, inp in enumerate(inputs):
        padded_sentences = [np.pad(s, (0, max_sen_len - sen_lens[i][j]), 'constant', constant_values=0) for j, s in
                            enumerate(inp)]
        # trim array according to max allowed inputs
        if len(padded_sentences) > max_len:
            padded_sentences = padded_sentences[(len(padded_sentences) - max_len):]
            lens[i] = max_len
        padded_sentences = np.vstack(padded_sentences)
        padded_sentences = np.pad(padded_sentences, ((0, max_len - lens[i]), (0, 0)), 'constant', constant_values=0)
        padded[i] = padded_sentences
    return padded


def create_embedding(word2vec, ivocab, embed_size):
    embedding = np.zeros((len(ivocab), embed_size))
    for i in range(len(ivocab)):
        word = ivocab[i]
        embedding[i] = word2vec[word]
    return embedding


def load_input_data(config, split_sentences=False):
    word_map = {}
    reverse_map = {}

    train_file, test_file = get_babi_files(config.babi_id, config.babi_test_id)

    if config.word2vec:
        assert config.embed_size == 100
        word2vec = load_glove(config.embed_size)
    else:
        word2vec = {}

    # set word at index zero to be special token so padding with zeros is consistent
    process_word(word="<END>",
                 word2vec=word2vec,
                 vocab=word_map,
                 ivocab=reverse_map,
                 word_vector_size=config.embed_size,
                 to_return="index")

    print('process train inputs')
    train_data = process_input(train_file, config.floatX, word2vec, word_map, reverse_map, config.embed_size,
                               )
    print('process test inputs')
    test_data = process_input(test_file, config.floatX, word2vec, word_map, reverse_map, config.embed_size,
                              )

    if config.word2vec:
        assert config.embed_size == 100
        word_embedding = create_embedding(word2vec, reverse_map, config.embed_size)
    else:
        word_embedding = np.random.uniform(-config.embedding_init, config.embedding_init,
                                           (len(reverse_map), config.embed_size))

    inputs, questions, answers, input_masks = train_data if config.train_mode else test_data
    input_vectors, sen_vectors, max_sen_len = get_sentence_vectors(inputs)
    max_mask_len = max_sen_len

    q_vectors = get_input_vectors(questions)

    max_q_len = np.max(q_vectors)
    max_input_len = min(np.max(input_vectors), config.max_allowed_inputs)

    # pad out arrays to max
    inputs = pad_inputs(inputs, input_vectors, max_input_len, sen_vectors, max_sen_len)
    input_masks = np.zeros(len(inputs))

    questions = pad_inputs(questions, q_vectors, max_q_len)
    answers = np.stack(answers)

    if config.train_mode:
        train = questions[:config.train_split], inputs[:config.train_split], q_vectors[:config.train_split], input_vectors[
                                                                                                       :config.train_split], input_masks[
                                                                                                                           :config.train_split], answers[
                                                                                                                                               :config.train_split]

        valid = questions[config.train_split:], inputs[config.train_split:], q_vectors[config.train_split:], input_vectors[
                                                                                                       config.train_split:], input_masks[
                                                                                                                           config.train_split:], answers[
                                                                                                                                               config.train_split:]
        return train, valid, word_embedding, max_q_len, max_input_len, max_mask_len, len(word_map), reverse_map

    else:
        test = questions, inputs, q_vectors, input_vectors, input_masks, answers
        return test, word_embedding, max_q_len, max_input_len, max_mask_len, len(word_map), reverse_map
