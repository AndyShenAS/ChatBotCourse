# coding:utf-8

import sys
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

import jieba
from jieba import analyse

def segment(input, output):
    input_file = open(input, "r")
    output_file = open(output, "w")
    while True:
        line = input_file.readline()
        if line:
            line = line.strip()
            seg_list = jieba.cut(line)
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

def segment_text(text):
    line = text.strip()
    seg_list = jieba.cut(line)
    # segments = ""
    segments = []
    for str in seg_list:
        segments.append(str)
    return segments





if __name__ == '__main__':
    if 3 != len(sys.argv):
        print("Usage: ", sys.argv[0], "input output")
        sys.exit(-1)
    segment(sys.argv[1], sys.argv[2]);
