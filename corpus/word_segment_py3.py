# coding:utf-8

import sys
import re
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

import jieba
from jieba import analyse

# def segment(inputs = ['./data/gossip.txt','./data/xiaohuangji.txt','../../../movieSubtitle/subtitle.corpus'], output = './data/corpus.segment.big'):
def segment(inputs = ['./data/gossip.txt','./data/xiaohuangji.txt','../../../movieSubtitle/subtitle.corpus'], output = './data/corpus.segment.huge'):
    output_file = open(output, "w")
    count = 0
    for input in inputs:
        print('file: ',input)        
        input_file = open(input, "r")
        print('file: ',input)
        while True:
            line = input_file.readline()
            if line:
                line = line.strip()
                if input == './data/gossip.txt':
                    # line_pair = line.split('#')
                    line_pair = re.split(r'#!|#',line)
                    line_pair.remove(line_pair[0])
                elif input == './data/xiaohuangji.txt':
                    line_pair = line.split('|')
                else:
                    line_pair = [line]
                while '' in line_pair:
                    line_pair.remove('')
                # line_question = line_pair[0]
                # line_answer = line_pair[1]
                for line_qa in line_pair:
                    count += 1
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
    print('segment ',count,' lines')
    output_file.close()

if __name__ == '__main__':
    # if 3 != len(sys.argv):
    #     print("Usage: ", sys.argv[0], "input output")
    #     sys.exit(-1)
    # segment(sys.argv[1], sys.argv[2]);
    segment();
