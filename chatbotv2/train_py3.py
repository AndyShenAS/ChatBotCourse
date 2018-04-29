# coding: utf-8
import pandas as pd
import numpy as np
import re
import time
import collections
import os
from bz2 import BZ2File
from io import open
from collections import Counter

import sys
import math
import tflearn
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
import chardet
import struct
sys.path.append('../')
from word_segment_py3 import segment,segment_text

import tensorflow as tf
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
print(('TensorFlow Version: {}'.format(tf.__version__)))

# sometimes we can use natural language toolkit to clean the text
import nltk
# if get an error not found some packages, use nltk.download()
# nltk.download()

from nltk import word_tokenize, sent_tokenize

# configurations of data path
# please put the corpus at this path
# data_path = 'data/output1.bz2'
# output_path = 'models/model1'

question_seqs = []
answer_seqs = []

def init_seq(input_file = './corpus.segment.pair'):
    """读取切好词的文本文件，加载全部词序列
    """
    file_object = open(input_file, 'r')
    vocab_dict = {}
    while True:
        question_seq = []
        answer_seq = []
        line = file_object.readline()
        if line:
            line_pair = line.split('|')
            line_question = line_pair[0]
            line_answer = line_pair[1]
            line_question = line_question.strip()
            line_answer = line_answer.strip()
            for word in line_question.split(' '):
                question_seq.append(word)
            for word in line_answer.split(' '):
                answer_seq.append(word)
        else:
            break
        question_seqs.append(question_seq)
        answer_seqs.append(answer_seq)
    file_object.close()

init_seq()
X = question_seqs
y = answer_seqs
# for i in range(30):
#     print(i)
#     print(X[i])
#     print(y[i])
y_clean = y
X_clean = X



word_counts = {}
def word_count(word_counts, text):
    for sentence in text:
        for word in sentence:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1

word_count(word_counts, X_clean)
word_count(word_counts, y_clean)
print("vocab size: ", len(word_counts))
count = 0
# for each in word_counts:
#     if count >29:
#         break
#     print(each,": ",word_counts[each])
#     count += 1

max_w = 50
float_size = 4
word_vec_dim = 200
word_vector_dict = {}
def load_vectors(input = "../corpus/data/vectors.bin"):
    """从vectors.bin加载词向量，返回一个word_vector_dict的词典，key是词，value是200维的向量
    """
    print("begin load vectors")

    input_file = open(input, "rb")

    # 获取词表数目及向量维度
    words_and_size = input_file.readline() # 这个.decode() 把二进制解码成unicode
    words_and_size = words_and_size.decode()
    words_and_size = words_and_size.strip()
    words = int(words_and_size.split(' ')[0])
    size = int(words_and_size.split(' ')[1])
    print("words =", words)
    print("size =", size)

    for b in range(0, words):
        a = 0
        word = bytes('','utf-8')
        # bytes("python", 'ascii')

        # 读取一个词
        while True:
            c = input_file.read(1)
            # c = c.decode()
            # print(c)
            word = word + c
            if False == c or c == bytes(' ','utf-8'):
                break
            if a < max_w and c != bytes('\n','utf-8'):
                a = a + 1
        word = word.decode()
        # print(word)
        word = word.strip()
        vector = []
        for index in range(0, size):
            m = input_file.read(float_size)
            (weight,) = struct.unpack('f', m)
            vector.append(float(weight))
        # 将词及其对应的向量存到dict中
        if word in word_counts:
            word_vector_dict[word] = vector[0:word_vec_dim]
    input_file.close()
    print("load vectors finish")
load_vectors()
embeddings_index = word_vector_dict

missing_words = 0
# threshold = 5
# add the words to this string and print
missing_words_list = []

for word, count in word_counts.items():
    if word not in embeddings_index:
        missing_words += 1
        missing_words_list.append(word)


missing_ratio = round(missing_words/len(word_counts), 4)*100
print("number of missing words: ", missing_words)
print("missing_ratio: ",missing_ratio)
# to see what words not covered by our word embeddings
print([word + ": " + str(word_counts[word]) for word in missing_words_list])

### word to int
word_to_int = {}

# 8 special texts at beginning
special_codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]
for code in special_codes:
    word_to_int[code] = len(word_to_int)

value = len(word_to_int)
for word, count in word_counts.items():
    if word in embeddings_index:
        word_to_int[word] = value
        value += 1

int_to_word = {}
for word, value in word_to_int.items():
    int_to_word[value] = word

usage_ratio = round(1.0 * len(word_to_int) / len(word_counts), 4)
print("Number of word we will use", len(word_to_int))
print("Percentage of words we will use: {}%".format(usage_ratio * 100))

# ### Select word embeddings which only appears in our samples to build our own word embedding
embedding_dim = 200 + len(special_codes)
n_words = len(word_to_int)

count = 0
word_embedding_matrix = np.zeros((n_words, embedding_dim), dtype=np.float32)
for word, i in word_to_int.items():
    if word in embeddings_index:
        temp = np.zeros((len(special_codes)))
        word_embedding_matrix[i] = np.concatenate([embeddings_index[word], temp], axis=0)
    elif word in special_codes:
        temp = np.zeros((embedding_dim))
        temp[embedding_dim - 1 - special_codes.index(word)] = 1
        word_embedding_matrix[i] = temp
    #     else:
#         # if not in the embedding matrix, initialize this vector by random number
#         new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
#         embeddings_index[word] = new_embedding
#         word_embedding_matrix[i] = new_embedding

print("the total number of words in word_embedding_matrix:", len(word_embedding_matrix))

def convert_to_ints(text, n_words, n_unk, eos=False):
    ints = []
    for sentence in text:
        sentence_ints = []
        for word in sentence:
            n_words += 1
            if word in word_to_int:
                sentence_ints.append(word_to_int[word])
            else:
                sentence_ints.append(word_to_int["<UNK>"])
                n_unk += 1
        if eos:
            sentence_ints.append(word_to_int["<EOS>"])
        ints.append(sentence_ints)
    return ints, n_words, n_unk

n_words = 0
n_unk = 0

int_y, n_words, n_unk = convert_to_ints(y_clean, n_words, n_unk)
int_X, n_words, n_unk = convert_to_ints(X_clean, n_words, n_unk, eos=True)

unk_percent = round(1.0 * n_unk / n_words, 4) * 100

print("Total number of words:", n_words)
print("Total number of UNKs:", n_unk)
print("Percent of words that are UNK: {}%".format(unk_percent))

def get_sentences_lengths(text):
    lengths = []
    for sentence in text:
        lengths.append(len(sentence))
    return pd.DataFrame(lengths, columns=['counts'])

lengths_y = get_sentences_lengths(int_y)
lengths_X = get_sentences_lengths(int_X)

#### plot to see the distribution of sentences length

import matplotlib.pyplot as plt
# get_ipython().magic(u'matplotlib inline')
# plt.hist(lengths_X['counts'], 20)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.show()
#
# plt.hist(lengths_y['counts'], 20)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.show()

print(lengths_y.describe())
print(lengths_X.describe())

def unk_counter(sentence):
    unk_count = 0
    for word in sentence:
        if word == word_to_int["<UNK>"]:
            unk_count += 1
    return unk_count

# parameters for batch parser
# set the length according to the 90% and 95% quantile
max_X_length = 10
max_y_length = 10
min_length = 1

# drop the sample if we got too much unknown words
unk_X_threshold = 1
unk_y_threshold = 0

def invalid_sample(code_list, invalid_ratio=0.75):
    count = 0
    for code in code_list:
        if code < len(special_codes):
            count += 1
    if 1.0 * count / len(code_list) > invalid_ratio:
        return True
    return False

sorted_y = []
sorted_X = []

for length in range(1, max_X_length):
    for i, sentence in enumerate(int_y):
        if (len(int_y[i]) >= min_length and
            len(int_y[i]) <= max_y_length and
            unk_counter(int_y[i]) <= unk_y_threshold and
            unk_counter(int_X[i]) <= unk_X_threshold and
            length == len(int_X[i])
           ):
            # if invalid_sample(int_X[i]):
            #     continue
            sorted_y.append(int_y[i])
            sorted_X.append(int_X[i])

print('number of sorted input', len(sorted_X))
print('number of sorted output', len(sorted_y))
print(sorted_X[:100])

print(word_embedding_matrix.shape)
print(word_embedding_matrix[-2])



#
#
#
#
# SET Hyperparams at first !!!
learning_rate = 0.001
learning_rate_decay = 0.95
min_learning_rate = 0.00005
epochs = 100
batch_size = 32
keep_probability = 0.75
# 1 - GradientDescentOptimizer
# 2 - AdamOptimizer
# 3 - RMSPropOptimizer
model_optimizer = 2

# Hyperparams for cells
# 1 - Basic RNN
# 2 - GRU
# 3 - LSTM
encoder_cell_type = 3
decoder_cell_type = 3
rnn_dim = 512
encoder_forget_bias = 1.0
decoder_forget_bias = 1.0

# 1 - tf.random_uniform_initializer
# 2 - tf.truncated_normal_initializer
# 3 - tf.orthogonal_initializer
initializer_type = 3

# 1 - Relu
# 2 - tanh
activation = None
num_layers = 1

# Hyperparams for attentions
# 1 - tf.contrib.seq2seq.BahdanauAttention()
# 2 - tf.contrib.seq2seq.LuongAttetion()
# 3 - no attention
attention_type = 1

# others
# gradient clipping
# the steep cliffs commonly occur in recurrent neural networks in the area where
# the recurrent network behaves approximately linearly, SGD or other methods without
# gradient clipping overshoots the landscape minimum, while the one with gradient
# clipping descents into the minimum.
# 1 - tf.clip_by_values(tensor, clip_value_min, clip_value_max)
# value less than clip_value_min and greater than clip_value_max will constrained
# 2 - tf.clip_by_norm(tensor, clip_norm, axes = None, name = None)
# tensor = t * clip_norm / l2norm(t)
model_gradient_clipping = 1
clip_value_min = -3
clip_value_max = 3
clip_norm = 5

log_dir = '/tmp/tensorflow/my_seq2seq_logs'

def model_inputs():

    input_data = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    lr = tf.placeholder(tf.float32, name='learning_rate')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    y_length = tf.placeholder(tf.int32, (None,), name='y_length')
    max_y_length = tf.reduce_max(y_length, name='max_decoder_len')
    X_length = tf.placeholder(tf.int32, (None,), name='X_length')

    return input_data, targets, lr, keep_prob, y_length, max_y_length, X_length

def process_decoding_input(target_data, word_to_int, batch_size):

    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    decoder_input = tf.concat([tf.fill([batch_size, 1], word_to_int['<GO>']), ending], axis=1)
    return decoder_input

def encoding_layer(rnn_dim, sequence_length, num_layers, rnn_inputs, keep_prob):

    # choose the initializer to use in cells
    if initializer_type == 1:
        initializer = tf.random_uniform_initializer(1.0, 1.0, seed=2)
    elif initializer_type == 2:
        initializer = tf.truncated_normal_initializer(1.0, 1.0, seed=2)
    else:
        initializer = tf.orthogonal_initializer(gain=1.0, seed=2)

    # choose the cell type to use
    if encoder_cell_type == 1:
        tf_cell = tf.contrib.rnn.RNNCell(rnn_dim)
    elif encoder_cell_type == 2:
        tf_cell = tf.contrib.rnn.GRUCell(rnn_dim,
            kernel_initializer=initializer,
            activation=activation)
    else:
        tf_cell = tf.contrib.rnn.LSTMCell(rnn_dim,
            initializer=initializer,
            forget_bias=1.0,
            activation=activation)

    # multilayered bidirecitonal RNN
    # https://stackoverflow.com/questions/44483560/multilayered-bi-directional-encoder-for-seq2seq-in-tensorflow
    next_inputs = rnn_inputs
    output_list = []
    for layer in range(num_layers):
        with tf.variable_scope('encoder_{}'.format(layer)):
            cell_fw = tf_cell
            # add dropout wrapper
            cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw,
                                input_keep_prob = keep_prob)
            cell_bw = tf_cell
            cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw,
                                input_keep_prob = keep_prob)

            # add bidirectional wrapper
            (encoder_output_fw, encoder_output_bw), encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw,
                                                                    cell_bw,
                                                                    next_inputs,
                                                                    sequence_length,
                                                                    dtype=tf.float32)
            # update the next_inputs from the output of current layer
            next_inputs = tf.concat([encoder_output_fw,encoder_output_bw], axis=2)
            output_list.append(next_inputs)

    # only take last one as encoder output
    # encoder_output = next_inputs
    # take all the outputs as encoder output
    if num_layers == 1:
        encoder_output = next_inputs
    else:
        encoder_output = tf.concat(output_list, axis=2)

    return encoder_output, encoder_state

# the decoding layer used in training
def training_decoding_layer(decoder_embed_input, y_length, decoder_cell, initial_state,
                            output_layer, vocab_size, max_y_length):
    '''Create the training logits'''
    # 3 steps for decoding layer:
    # helper: use argmax, go through embeddings
    # basic decoder: decoder
    # dynamic_decode: perform dynamic decoding with the decoder chose
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                        sequence_length=y_length,
                                                        time_major=False)
    # time major means:
    # If time_major == False (default), this must be a Tensor of shape: [batch_size, max_time, ...], or a nested tuple of such elements.
    # If time_major == True, this must be a Tensor of shape: [max_time, batch_size, ...], or a nested tuple of such elements.
    training_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                       helper=training_helper,
                                                       initial_state=initial_state,
                                                       output_layer=output_layer)

    # impute_finished: Python boolean. If True, then states for batch entries which are marked as finished get copied through and the corresponding outputs get zeroed out. This causes some slowdown at each time step, but ensures that the final state and outputs have the correct values and that backprop ignores time steps that were marked as finished.
    # set zero when finished and improve the performance
    training_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=training_decoder,
                                                           output_time_major=False,
                                                           impute_finished=True,
                                                           maximum_iterations=max_y_length)
    return training_logits

# the decoding layer used in using
def inference_decoding_layer(embeddings, start_token, end_token, decoder_cell, initial_state,
                             output_layer, max_y_length, batch_size):
    '''Create the inference logits'''
    # <GO> symbol
    start_tokens = tf.tile(tf.constant([start_token], dtype=tf.int32), [batch_size], name='start_tokens')
    # A helper for use during inference
    # use the argmax of the output and passes the result through an embedding layer to get the next input
    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=embeddings,
                                                                start_tokens=start_tokens,
                                                                end_token=end_token)
    # Basic sampling decoder
    inference_decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell,
                                                        helper=inference_helper,
                                                        initial_state=initial_state,
                                                        output_layer=output_layer)

    inference_logits, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=inference_decoder,
                                                            output_time_major=False,
                                                            impute_finished=True,
                                                            maximum_iterations=max_y_length)

    return inference_logits

def decoding_layer(decoder_embed_input, embeddings, encoder_output, encoder_state,
                   vocab_size, X_length, y_length, max_y_length, rnn_dim, word_to_int,
                   keep_prob, batch_size, num_layers):
    '''Create the decoding cell and attention for the training and inference decoding layers'''
    # choose the initializer to use in cells
    if initializer_type == 1:
        initializer = tf.random_uniform_initializer(1.0, 1.0, seed=2)
    elif initializer_type == 2:
        initializer = tf.truncated_normal_initializer(1.0, 1.0, seed=2)
    else:
        initializer = tf.orthogonal_initializer(gain=1.0, seed=2)

    # choose the cell type to use
    if decoder_cell_type == 1:
        tf_cell = tf.contrib.rnn.RNNCell(rnn_dim)
    elif decoder_cell_type == 2:
        tf_cell = tf.contrib.rnn.GRUCell(rnn_dim,
            kernel_initializer=initializer,
            activation=activation)
    else:
        tf_cell = tf.contrib.rnn.LSTMCell(rnn_dim,
            initializer=initializer,
            forget_bias=1.0,
            activation=activation)

    # create cells for decoder
    # for layer in range(num_layers):
    with tf.variable_scope('decoder'):
        lstm = tf_cell
        decoder_cell = tf.contrib.rnn.DropoutWrapper(lstm,
                                                     input_keep_prob = keep_prob)

    output_layer = Dense(vocab_size,
                         kernel_initializer = tf.truncated_normal_initializer(mean = 0.0, stddev=0.25))

    # # implements Bahdanau-style (additive) attetion
    if attention_type == 1:
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(rnn_dim,
                                                         encoder_output,
                                                         X_length,
                                                         normalize=False,
                                                         name='BahdanauAttention')
    else:
        #implements Luong-style(multiplication) attention
        attn_mech = tf.contrib.seq2seq.LuongAttention(rnn_dim,
                                                      encoder_output,
                                                      X_length,
                                                      name='LuongAttention')
    # Wraps RNNCells with attention
    decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell,
                                                       attn_mech,
                                                       rnn_dim)

    # initial_cell_state = the ending state of encoder
    initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=encoder_state[0])

    with tf.variable_scope("decode"):
        training_logits = training_decoding_layer(decoder_embed_input,
                                                  y_length,
                                                  decoder_cell,
                                                  initial_state,
                                                  output_layer,
                                                  vocab_size,
                                                  max_y_length)
    with tf.variable_scope("decode", reuse=True):
        inference_logits = inference_decoding_layer(embeddings,
                                                    word_to_int['<GO>'],
                                                    word_to_int['<EOS>'],
                                                    decoder_cell,
                                                    initial_state,
                                                    output_layer,
                                                    max_y_length,
                                                    batch_size)

    return training_logits, inference_logits

def seq2seq_model(input_data, target_data, keep_prob, X_length, y_length, max_y_length,
                  vocab_size, rnn_dim, num_layers, word_to_int, batch_size):
    '''Use the previous functions to create the training and inference logits'''
    # The whole flow graph of the seq2seq model
    # embedding --> encoding layer(bidirectional lstm) --> output -->
    # decoding layer (dynamic decoder with bahdanau style (additive) ttention)

    # Use Numberbatch's embeddings and the newly created ones as our embeddings
    embeddings = word_embedding_matrix

    # convert the input int to vectors
    encoder_embed_input = tf.nn.embedding_lookup(embeddings, input_data)
    # encoding layer
    encoder_output, encoder_state = encoding_layer(rnn_dim,
                                                   X_length,
                                                   num_layers,
                                                   encoder_embed_input,
                                                   keep_prob)

    decoder_input = process_decoding_input(target_data, word_to_int, batch_size)
    decoder_embed_input = tf.nn.embedding_lookup(embeddings, decoder_input)

    training_logits, inference_logits  = decoding_layer(decoder_embed_input,
                                                        embeddings,
                                                        encoder_output,
                                                        encoder_state,
                                                        vocab_size,
                                                        X_length,
                                                        y_length,
                                                        max_y_length,
                                                        rnn_dim,
                                                        word_to_int,
                                                        keep_prob,
                                                        batch_size,
                                                        num_layers)

    return training_logits, inference_logits

def pad_sentence_batch(sentences):
    max_sentence = max([len(sentence) for sentence in sentences])
    return [sentence + [word_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentences]


def reverse_sentence_batch(sentences):
    # reverse the inputs
    return [list(reversed(sentence)) for sentence in sentences]


def get_batches(summaries, texts, batch_size):
    """Batch summaries, texts, and the lengths of their sentences together"""
    for batch_i in range(0, len(texts)//batch_size):
        start_i = batch_i * batch_size
        summaries_batch = summaries[start_i:start_i + batch_size]
        texts_batch = texts[start_i:start_i + batch_size]
        pad_summaries_batch = np.array(pad_sentence_batch(summaries_batch))
        pad_texts_batch = np.array(reverse_sentence_batch(pad_sentence_batch(texts_batch)))

        # Need the lengths for the _lengths parameters
        pad_summaries_lengths = []
        for summary in pad_summaries_batch:
            pad_summaries_lengths.append(len(summary))

        pad_texts_lengths = []
        for text in pad_texts_batch:
            pad_texts_lengths.append(len(text))

        yield pad_summaries_batch, pad_texts_batch, pad_summaries_lengths, pad_texts_lengths

### Building the graph ###
print("Building the model")

def train():
    global learning_rate
    train_graph = tf.Graph()

    with train_graph.as_default():
        input_data, targets, lr, keep_prob, y_length, max_y_length, X_length = model_inputs()

        training_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),
                                                          targets,
                                                          keep_prob,
                                                          X_length,
                                                          y_length,
                                                          max_y_length,
                                                          len(word_to_int) + 1,
                                                          rnn_dim,
                                                          num_layers,
                                                          word_to_int,
                                                          batch_size)

        training_logits = tf.identity(training_logits.rnn_output, 'logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

        # Create the weights for sequence_loss
        masks = tf.sequence_mask(y_length, max_y_length, dtype=tf.float32, name='mask')

        with tf.name_scope("optimization"):
            cost = tf.contrib.seq2seq.sequence_loss(training_logits, targets, masks)

            # Here we can choose optimizer used in the model
            if model_optimizer == 1:
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            elif model_optimizer == 2:
                optimizer = tf.train.AdamOptimizer(learning_rate)
            else:
                optimizer = tf.train.RMSPropOptimizer(learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            if model_gradient_clipping == 1:
                capped_gradients = [(tf.clip_by_value(grad, clip_value_min, clip_value_max), var) for grad, var in gradients if grad is not None]
            else:
                capped_gradients = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grfadients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
            ######################################
            tf.summary.scalar('loss',cost)
            #########################################

    print("Model is built")

    # writer = tf.summary.FileWriter("./demo/graph")
    # writer.add_graph(sess.graph)

    # cut the dataset to training set
    # start = 200000
    # # end = start + 50000
    # sorted_y_short = sorted_y[start:end]
    # sorted_X_short = sorted_X[start:end]
    sorted_y_short = sorted_y
    sorted_X_short = sorted_X

    print(("The shortest X length:", len(sorted_X_short[0])))
    print(("The longest X length:",len(sorted_X_short[-1])))

    # parameters in training:
    display_step = 20 # Check training loss after every 20 batches
    stop_early = 0
    # If the update loss does not decrease in num_to_stop consecutive update checks, stop training
    num_to_stop = 7

    per_epoch = 3

    # update checking
    update_check = (len(sorted_X_short)//batch_size//per_epoch)-1
    update_loss = 0
    batch_loss = 0
    y_update_loss = [] # Record the update losses for saving improvements in the model

    checkpoint = "./models/best_model.ckpt"

    config = tf.ConfigProto()
    config.gpu_options.allocator_type ='BFC'
    config.gpu_options.per_process_gpu_memory_fraction = 0.80

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        # If we want to continue training a previous session
        #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
        #loader.restore(sess, checkpoint)
        ####################################################
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        ####################################################

        for epoch_i in range(1, epochs+1):
            update_loss = 0
            batch_loss = 0
            for batch_i, (y_batch, X_batch, y_lengths, X_lengths) in enumerate(
                    get_batches(sorted_y_short, sorted_X_short, batch_size)):
                start_time = time.time()
                _, loss, summary_str = sess.run(
                    [train_op, cost, merged_summary_op],
                    {input_data: X_batch,
                     targets: y_batch,
                     lr: learning_rate,
                     y_length: y_lengths,
                     X_length: X_lengths,
                     keep_prob: keep_probability})

                batch_loss += loss
                update_loss += loss
                end_time = time.time()
                batch_time = end_time - start_time

                ############################################
                # summary_str = sess.run(merged_summary_op)

                summary_writer.add_summary(summary_str, epoch_i)
                ###################################################


                if batch_i % display_step == 0 and batch_i > 0:
                    print(('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                          .format(epoch_i,
                                  epochs,
                                  batch_i,
                                  len(sorted_X_short) // batch_size,
                                  batch_loss / display_step,
                                  batch_time*display_step)))
                    batch_loss = 0

                if batch_i % update_check == 0 and batch_i > 0:
                    print(("Average loss for this update:", round(update_loss/update_check,3)))
                    y_update_loss.append(update_loss)

                    # If the update loss is at a new minimum, save the model
                    if update_loss <= min(y_update_loss):
                        print('New Record!')
                        stop_early = 0
                        saver = tf.train.Saver()
                        saver.save(sess, checkpoint)

                    else:
                        print("No Improvement.")
                        stop_early += 1
                        if stop_early == num_to_stop:
                            break
                    update_loss = 0


            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            if stop_early == num_to_stop:
                print("Stopping Training.")
                break
##################################################################################

train()

##################################################################################







def text_to_seq(text):
    '''Prepare the text for the model'''

    # text = clean_text(text)
    temp_list = [word_to_int.get(word, word_to_int['<UNK>']) for word in segment_text(text)]
    return list(reversed(temp_list))

# >>> a={"a":1,"b":2}
# >>> b= ["a","b","c"]
# >>> c = [a.get(each,3) for each in b]
# >>> c
# [1, 2, 3]
# >>> b= ["a","b","c","d"]
# >>> c = [a.get(each,3) for each in b]
# >>> c
# [1, 2, 3, 3]


def predict():

    # Create your own review or use one from the dataset
    # input_sentence = "Do you like Joshua?"
    input_sentence = "世界上最美的人是谁"
    text = text_to_seq(input_sentence)
    # random = np.random.randint(0,len(clean_texts))
    # input_sentence = clean_texts[random]
    # text = text_to_seq(clean_texts[random])

    checkpoint = "./models/best_model.ckpt"

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        X_length = loaded_graph.get_tensor_by_name('X_length:0')
        y_length = loaded_graph.get_tensor_by_name('y_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        #Multiply by batch_size to match the model's input parameters
        answer_logits = sess.run(logits, {input_data: [text]*batch_size,
                                          y_length: [np.random.randint(35,40)],
                                          X_length: [len(text)]*batch_size,
                                          keep_prob: 1.0})[0]

    # Remove the paddings
    pad = word_to_int["<PAD>"]

    print(('Original Text:', input_sentence))

    print('\nText')
    print(('  Word Ids:    {}'.format([i for i in text])))
    print(('  Input Words: {}'.format(" ".join([int_to_word[i] for i in text]))))

    print('\nSummary')
    print(('  Word Ids:       {}'.format([i for i in answer_logits if i != pad])))
    print(('  Response Words: {}'.format(" ".join([int_to_word[i] for i in answer_logits if i != pad]))))

#####################################################
predict()
########################################################




def save_others():

    import pickle
    f = open('int_to_word.pkl', 'wb')
    pickle.dump(int_to_word, f)
    f.close()

    f = open('word_to_int.pkl', 'wb')
    pickle.dump(word_to_int, f)
    f.close()
