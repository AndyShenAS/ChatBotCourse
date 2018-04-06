# -*- coding: utf-8 -*-
#smy smy
import sys
import math
import tflearn
import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn
import chardet
import numpy as np
import struct
sys.path.append('../')
from word_segment_py3 import segment

class WordVector(object):
    def __init__(self):
        self.question_seqs = []
        self.answer_seqs = []

        self.max_w = 50
        self.float_size = 4
        self.word_vector_dict = {}
        self.word_vec_dim = 200
        self.word_set = {}
        self.load_vectors()


    def load_word_set():
        file_object = open('./corpus.segment.pair', 'r')
        while True:
            line = file_object.readline()
            if line:
                line_pair = line.split('|')
                line_question = line_pair[0]
                line_answer = line_pair[1]
                for word in line_question.split(' '):
                    # for word in line_question.decode('utf-8').split(' '):   python 3中只有unicode str，所以把decode方法去掉了。你的代码中，f1已经是unicode str了，不用decode。
                    self.word_set[word] = 1
                for word in line_answer.split(' '):
                    self.word_set[word] = 1
            else:
                break
        file_object.close()

    def load_vectors(input = './vectors.bin'):
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
                if a < self.max_w and c != bytes('\n','utf-8'):
                    a = a + 1
            word = word.decode()
            # print(word)
            word = word.strip()

            vector = []
            for index in range(0, size):
                m = input_file.read(self.float_size)
                (weight,) = struct.unpack('f', m)
                vector.append(float(weight))

            # 将词及其对应的向量存到dict中

            if word in self.word_set:
                self.word_vector_dict[word] = vector[0:self.word_vec_dim]

        input_file.close()

        print("load vectors finish")

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
                for word in line_question.split(' '):
                    if word in self.word_vector_dict:
                        question_seq.append(self.word_vector_dict[word])
                for word in line_answer.split(' '):
                    if word in self.word_vector_dict:
                        answer_seq.append(self.word_vector_dict[word])
            else:
                break
            self.question_seqs.append(question_seq)
            self.answer_seqs.append(answer_seq)
        file_object.close()

    def vector_sqrtlen(vector):
        len = 0
        for item in vector:
            len += item * item
        len = math.sqrt(len)
        return len

    def vector_cosine(v1, v2):
        if len(v1) != len(v2):
            sys.exit(1)
        sqrtlen1 = self.vector_sqrtlen(v1)
        sqrtlen2 = self.vector_sqrtlen(v2)
        value = 0
        for item1, item2 in zip(v1, v2):
            value += item1 * item2
        return value / (sqrtlen1*sqrtlen2)


    def vector2word(vector):
        max_cos = -10000
        match_word = ''
        for word in self.word_vector_dict:
            v = self.word_vector_dict[word]
            cosine = self.vector_cosine(vector, v)
            if cosine > max_cos:
                max_cos = cosine
                match_word = word
        return (match_word, max_cos)

    def word2vector(word):
        if not isinstance(word, str):
            print("Exception: error word not unicode")
            sys.exit(1)
        if word in self.word_vector_dict:
            return self.word_vector_dict[word]
        else:
            return None
