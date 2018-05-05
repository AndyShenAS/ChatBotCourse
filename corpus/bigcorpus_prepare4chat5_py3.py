# coding:utf-8

import sys
import re
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

import jieba
from jieba import analyse

def segment(inputs = ['./data/gossip.txt','./data/xiaohuangji.txt'], output = ['./data/question.big','./data/answer.big']):
    output_file_question = open(output[0], "w")
    output_file_answer = open(output[1], "w")
    for input in inputs:
        input_file = open(input, "r")
        while True:
            line = input_file.readline()
            if line:
                line = line.strip()
                if input == './data/gossip.txt':
                    # line_pair = line.split('#')
                    line_pair = re.split(r'#!|#',line)
                    line_pair.remove(line_pair[0])
                else:
                    line_pair = line.split('|')
                while '' in line_pair:
                    line_pair.remove('')
                print(line_pair)
                if len(line_pair) != 2:
                    continue
                line_question = line_pair[0]
                line_answer = line_pair[1]
                output_file_question.write(line_question+'\n')
                output_file_answer.write(line_answer+'\n')
                # for line_qa in line_pair:
                #     seg_list = jieba.cut(line_qa)
                #     # segments = ""
                #     segments = "GO_ID"
                #     for str in seg_list:
                #         segments = segments + " " + str
                #     segments = segments + " EOS_ID"+"\n"

            else:
                break
        input_file.close()
    output_file_question.close()
    output_file_answer.close()
    print('finish.....')

if __name__ == '__main__':
    # if 3 != len(sys.argv):
    #     print("Usage: ", sys.argv[0], "input output")
    #     sys.exit(-1)
    # segment(sys.argv[1], sys.argv[2]);
    segment();
