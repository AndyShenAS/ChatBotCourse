# coding:utf-8

import sys
# reload(sys)
# sys.setdefaultencoding( "utf-8" )

import jieba
from jieba import analyse

def segment(input, output):
    input_file = open(input, "r")
    output_file = open(output, "w")
    segments = ""
    while True:
        line = input_file.readline()
        if line:
            line = line.strip()
            seg_list = jieba.cut(line)
            segment = ''
            # segments = "GO_ID"
            for seg in seg_list:
                if seg != '\n':
                    segment += seg + " "
            segments += segment + "\n"
        else:
            break
    output_file.write(segments)
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
    # if 3 != len(sys.argv):
    #     print("Usage: ", sys.argv[0], "input output")
    #     sys.exit(-1)
    # segment(sys.argv[1], sys.argv[2])
    # segment('chatbotv5/samples/question.big.norepeat', 'chatbotv5/samples/question.big.norepeat.segment')
    # segment('chatbotv5/samples/answer.big.norepeat', 'chatbotv5/samples/answer.big.norepeat.segment')
    # segment('chatbotv5/samples/backup/question', 'chatbotv5/samples/backup/question.segment')
    # segment('chatbotv5/samples/backup/answer', 'chatbotv5/samples/backup/answer.segment')
    # segment('chatbotv2/data/question.all.norepeat', 'chatbotv2/data/question.all.norepeat.segment')
    # segment('chatbotv2/data/answer.all.norepeat', 'chatbotv2/data/answer.all.norepeat.segment')
    # segment('./chatbotv5/samples/answer', './chatbotv5/samples/answer.segment')
    segment('./chatbotv5/samples/question', './chatbotv5/samples/question.segment')
