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
            # weight = np.array(weight)
            # print(type(weight))
            vector[index] = weight

        # 将词及其对应的向量存到dict中
        word_vector[word] = vector

    input_file.close()

    print("load vectors finish")
    return word_vector

def pca_transfer(dic):
    words = []
    weights = []
    for k,v in dic.items():
        words.append(k)
        weights.append(v)
    test_weights = []
    test_words =[]
    test_weights.append(dic["你"])
    test_weights.append(dic["我"])
    test_weights.append(dic["我们"])
    test_weights.append(dic["他"])
    test_weights.append(dic["她"])
    # test_words.extend(["你","我","我们","他","她"])
    test_words.extend(["you","I","we","he","her"])
    min_max_scaler = preprocessing.MinMaxScaler()
    globalMean_minmax = min_max_scaler.fit_transform(test_weights)
    #PCA 降维
    pca = PCA(n_components=0.95)
    pca.fit(globalMean_minmax)
    newData_PCA = pca.transform(globalMean_minmax)
    #setting plt
    plt.xlim(xmax=6,xmin=-6)
    plt.ylim(ymax=6,ymin=-6)
    plt.xlabel("width")
    plt.ylabel("height")
    for i in range(len(test_words)):
        plt.text(newData_PCA[i][0],newData_PCA[i][1],str(test_words[i]), family='serif', style='italic', ha='right', wrap=True)
    # plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    plt.grid(True)
    plt.show()
    print(newData_PCA)
    print(words)
    print('PCA result......')
    print(pca.explained_variance_ratio_)
    print(pca.explained_variance_)
    print(pca.n_components_)



if __name__ == '__main__':
    if 2 != len(sys.argv):
        print("Usage: ", sys.argv[0], "./chatbotv2/vectors.bin")
        sys.exit(-1)
        # 提示怎么写参数的file = open('question',"w", encoding='utf-8')
    dic = load_vectors(sys.argv[1])
    pca_transfer(dic)






    # np.set_printoptions(threshold=np.NaN)
    # count = 0
    # for each in keys:
    #     count = count + 1
    #     if count<3:
    #         print(each)
    #         print(d[each])
