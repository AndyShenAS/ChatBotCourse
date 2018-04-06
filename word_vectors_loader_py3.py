# coding: utf-8

import sys
import struct
import math
import numpy as np
# import importlib
#
# importlib.reload(sys)
#
# # reload(sys)
# sys.setdefaultencoding( "utf-8" )

max_w = 50
float_size = 4

def load_vectors(input):
    print("begin load vectors")

    input_file = open(input, "rb")

    # 获取词表数目及向量维度
    words_and_size = input_file.readline()
    words_and_size = words_and_size.decode()
    words_and_size = words_and_size.strip()
    words = int(words_and_size.split(' ')[0])
    size = int(words_and_size.split(' ')[1])
    print("words =", words)
    print("size =", size)

    word_vector = {}

    for b in range(0, words):
        a = 0
        word = bytes('','utf-8')
        # 读取一个词
        while True:
            c = input_file.read(1)
            word = word + c
            if False == c or c == bytes(' ','utf-8'):
                break
            if a < max_w and c != bytes('\n','utf-8'):
                a = a + 1
        word = word.decode()
        # print(word)
        word = word.strip()

        # 读取词向量
        vector = np.empty([200])
        for index in range(0, size):
            m = input_file.read(float_size)
            (weight,) = struct.unpack('f', m)
            vector[index] = weight

        # 将词及其对应的向量存到dict中
        word_vector[word] = vector

    input_file.close()

    print("load vectors finish")
    return word_vector

def load_word_set():
    file_object = open('./chatbotv2/corpus.segment', 'r')
    word_set = {}
    while True:
        line = file_object.readline()
        if line:
            line = line.strip('\n')
            for word in line.split(' '):
                if word in word_set.keys():
                    word_set[word] += 1;
                else:
                    word_set[word] = 1
        else:
            break
    file_object.close()
    return word_set

if __name__ == '__main__':
    if 2 != len(sys.argv):
        print("Usage: ", sys.argv[0], "vectors.bin")
        sys.exit(-1)
        # 提示怎么写参数的file = open('question',"w", encoding='utf-8')
    d = load_vectors(sys.argv[1])
    keys = d.keys()
    word_set = load_word_set()
    wrSTR = ''
    count = 0
    for each in keys:
        # print(each)
        if each in word_set.keys():
            count = count + 1
            wrSTR += str(each)+ ': ' + str(word_set[each])+ '\n'
            if count<5:
                print(each)
                print(d[each])
        else:
            print(each)
            print(d[each])


    # print(wrSTR)
    print(count)
    print(len(d))
    print(len(word_set))
    file = open('./chatbotv2/keyWords.txt',"w", encoding='utf-8')
    file.write(wrSTR)
    file.close()
    # print(word_set)
    # 按照第1个元素降序排列
    # >> dic
    # {'a':3 , 'b':2 , 'c': 1}
    # >> sorted(dict2list(dic), key=lambda x:x[1], reverse=True) # 按照第1个元素降序排列
    # [('a', 3), ('b', 2), ('c', 1)]
    #
    wrSTR = ''
    for k,v in sorted(word_set.items(), key=lambda x:x[1], reverse=True):
        wrSTR += str(k)+ ': ' + str(v)+ '\n'
    file = open('./chatbotv2/allWords.txt',"w", encoding='utf-8')
    file.write(wrSTR)
    file.close()
    # print(d[u'真的'])
    # print(d)
