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
import random

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
import jieba

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

# question_path = '../chatbotv5/samples/question.big.segment'
# answer_path = '../chatbotv5/samples/answer.big.segment'
question_path = 'data/question.all.norepeat.segment'
answer_path = 'data/answer.all.norepeat.segment'
#一定要预处理，把重复问题去掉，不然loss一直降不下来！！！


def get_train_set():
    with open(question_path, 'r') as question_file:
        with open(answer_path, 'r') as answer_file:
            while True:
                question_seq = []
                answer_seq = []
                question = question_file.readline()
                answer = answer_file.readline()
                if question and answer:
                    line_question = question.strip()
                    line_answer = answer.strip()
                    for word in line_question.split(' '):
                        question_seq.append(word)
                    for word in line_answer.split(' '):
                        answer_seq.append(word)
                else:
                    break
                question_seqs.append(question_seq)
                answer_seqs.append(answer_seq)

get_train_set()

X = question_seqs
y = answer_seqs
# for i in range(30):
#     random_i = random.randint(0,len(X))
#     print(random_i)
#     print('question:',X[random_i])
#     print('answer:',y[random_i])
y_clean = y
X_clean = X

print('number of original input', len(X_clean))
print('number of original output', len(y_clean))
count = 0
for i in range(len(X_clean)):
    # if '最美' in X_clean[i] and '老婆' in y_clean[i]:
    if '出去' in X_clean[i] and '玩' in X_clean[i] :
    # if '寂寞' in X_clean[i] and '聊天' in y_clean[i]:
        count += 1
        print(count)
        print('ques:',X_clean[i])
        print('ans:',y_clean[i])



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
threshold = 8
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
print([word + ": " + str(word_counts[word]) for word in missing_words_list[0:100]])

### word to int
word_to_int = {}

# 8 special texts at beginning
special_codes = ["<UNK>", "<PAD>", "<EOS>", "<GO>"]
for code in special_codes:
    word_to_int[code] = len(word_to_int)

value = len(word_to_int)
for word, count in word_counts.items():
    if count >= threshold and word in embeddings_index:
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

# int_y, n_words, n_unk = convert_to_ints(y_clean, n_words, n_unk)
# int_X, n_words, n_unk = convert_to_ints(X_clean, n_words, n_unk, eos=True)
int_y, n_words, n_unk = convert_to_ints(y_clean, n_words, n_unk, eos=True)
int_X, n_words, n_unk = convert_to_ints(X_clean, n_words, n_unk)

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

# plt.hist(lengths_X['counts'], 200)
# plt.xlabel('Sequence Length')
# plt.ylabel('Frequency')
# plt.show()
#
# plt.hist(lengths_y['counts'], 200)
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
max_X_length = 12
max_y_length = 12
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


count_invalid = 0
for length in range(1, max_X_length):
    for i, sentence in enumerate(int_y):
        if (len(int_y[i]) >= min_length and
            len(int_y[i]) <= max_y_length and
            unk_counter(int_y[i]) <= unk_y_threshold and
            unk_counter(int_X[i]) <= unk_X_threshold and
            length == len(int_X[i])
           ):
            if invalid_sample(int_X[i]):
                # print('length:',length)
                # print('invalid question:',[int_to_word[id] for id in int_X[i]])
                # print('invalid answer:',[int_to_word[id] for id in int_y[i]])
                count_invalid += 1
                continue
            sorted_y.append(int_y[i])
            sorted_X.append(int_X[i])
print('count_invalid:',count_invalid)
print('number of sorted input', len(sorted_X))
print('number of sorted output', len(sorted_y))

# print(sorted_X[:100])
# print(sorted_y[:100])

print(word_embedding_matrix.shape)
# print(word_embedding_matrix[-2])

for i in range(30):
    random_i = random.randint(0,len(sorted_X))
    print(random_i)
    print('question:',[int_to_word[id] for id in sorted_X[random_i]])
    print('answer:',[int_to_word[id] for id in sorted_y[random_i]])



def save_others():

    import pickle
    f = open('int_to_word.pkl', 'wb')
    pickle.dump(int_to_word, f)
    f.close()

    f = open('word_to_int.pkl', 'wb')
    pickle.dump(word_to_int, f)
    f.close()
