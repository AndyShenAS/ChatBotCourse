# coding:utf-8

import sys
import re
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

import jieba
from jieba import analyse

def segment(inputs = ['./data/gossip.txt','./data/xiaohuangji.txt'], output = './data/corpus.segment.big'):
    output_file = open(output, "w")
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
                # line_question = line_pair[0]
                # line_answer = line_pair[1]
                for line_qa in line_pair:
                    seg_list = jieba.cut(line_qa)
                    # segments = ""
                    segments = "GO_ID"
                    for str in seg_list:
                        segments = segments + " " + str
                    segments = segments + " EOS_ID"+"\n"
                    output_file.write(segments)
            else:
                break
        input_file.close()
    output_file.close()

if __name__ == '__main__':
    # if 3 != len(sys.argv):
    #     print("Usage: ", sys.argv[0], "input output")
    #     sys.exit(-1)
    # segment(sys.argv[1], sys.argv[2]);
    segment();
