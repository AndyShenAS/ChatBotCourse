import sys
import struct
import math
import numpy as np
import  scipy.stats as stat
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import MeanShift, estimate_bandwidth

max_w = 50
float_size = 4


def load_vectors(input = './data/sentence_vectors.dic'):
    print("begin load vectors")

    input_file = open(input, "r")
    a = input_file.read()
    sentence_vector = eval(a)
    input_file.close()
    print("load vectors finish")
    # for each in sentence_vector:
    #     print(each,': \n')
    #     print(sentence_vector[each],'\n')
    return sentence_vector

def vector_sqrtlen(vector):
    len = 0
    for item in vector:
        len += item * item
    len = math.sqrt(len)
    return len

def vector_cosine(v1, v2):
    if len(v1) != len(v2):
        sys.exit(1)
    sqrtlen1 = vector_sqrtlen(v1)
    sqrtlen2 = vector_sqrtlen(v2)
    value = 0
    for item1, item2 in zip(v1, v2):
        value += item1 * item2
    return value / (sqrtlen1*sqrtlen2)

def eval_distance(dic):
    for sentence in dic:
        dist_dic = {}
        for sentence_v in dic:
            if sentence == sentence_v:
                continue
            dist = vector_cosine(dic[sentence],dic[sentence_v])
            dist_dic[sentence_v] = dist
        xx = sorted(dist_dic.items(),key = lambda x:x[1],reverse = True)
        print('distance between',sentence,'......')
        for each in xx:
            print(each[0],each[1])


def pca_transfer(dic):
    sentences = []
    weights = []
    for k,v in dic.items():
        sentences.append(k)
        weights.append(v)
    test_weights = []
    test_sentences =[]
    test_weights.append(dic["我喜欢你"])
    test_weights.append(dic["我爱你"])
    test_weights.append(dic["我讨厌你"])
    test_weights.append(dic["我恨你"])
    test_weights.append(dic["你喜欢我"])
    test_weights.append(dic["你爱我"])
    test_weights.append(dic["你讨厌我"])
    test_weights.append(dic["你恨我"])
    test_sentences.extend(["I like you","I love you","I dislike you","I hate you","You like me","You love me","You dislike me","You hate me"])
    # test_words.extend(["man","woman","king","queen","princess","prince","male","female"])
    min_max_scaler = preprocessing.MinMaxScaler()
    globalMean_minmax = min_max_scaler.fit_transform(test_weights)
    #PCA 降维
    pca = PCA(n_components=0.95)
    pca.fit(globalMean_minmax)
    newData_PCA = pca.transform(globalMean_minmax)
    #setting plt
    # plt.xlim(xmax=6,xmin=-6)
    # plt.ylim(ymax=6,ymin=-6)
    plt.xlabel("width",fontsize=18)
    plt.ylabel("height",fontsize=18)
    for i in range(len(test_sentences)):
        plt.text(newData_PCA[i][0],newData_PCA[i][1]+0.2,str(test_sentences[i]), family='serif', style='italic', ha='right', wrap=True, fontsize=18)
        plt.scatter(newData_PCA[i][0],newData_PCA[i][1], color='', marker='o', edgecolors='r', s=60)
    # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.grid(True)
    plt.show()
    print(newData_PCA)
    # print(words)
    print('PCA result......')
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.n_components_)



if __name__ == '__main__':
    # if 2 != len(sys.argv):
    #     print("Usage: ", sys.argv[0], "./chatbotv2/vectors.bin")
    #     sys.exit(-1)
        # ./data/vectors.bin
        # 提示怎么写参数的file = open('question',"w", encoding='utf-8')
    dic = load_vectors('./data/sentence_vectors.dic')
    # pca_transfer(dic)
    eval_distance(dic)






    # np.set_printoptions(threshold=np.NaN)
    # count = 0
    # for each in keys:
    #     count = count + 1
    #     if count<3:
    #         print(each)
    #         print(d[each])
