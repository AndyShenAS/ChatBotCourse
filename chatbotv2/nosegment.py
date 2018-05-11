# coding:utf-8
import sys
import jieba


question_path = './data/test.question'
answer_path = './data/test.answer'

question_path_out = './data/test.question.nosegment'
answer_path_out = './data/test.answer.nosegment'




ques_set = []
ans_set = []

def get_train_set(question_path = question_path ,answer_path = answer_path):
    count = 0
    with open(question_path, 'r') as question_file:
        with open(answer_path, 'r') as answer_file:
            while True:
                question = question_file.readline()
                answer = answer_file.readline()
                count += 1
                if question and answer:
                    # question = question.replace(" ", "")
                    # answer = answer.replace(" ", "")
                    ques_set.append("".join([word for word in question.split(' ')]))
                    ans_set.append("".join([word for word in answer.split(' ')]))

                else:
                    break
    # print('original num: ', count)

def save_train_set():
    question_file = open(question_path_out, 'w')
    answer_file = open(answer_path_out, 'w')
    for i in range(len(ques_set)):
        # question_file.write(ques_set[i]+'\n')
        # answer_file.write(ans_set[i]+'\n')
        question_file.write(ques_set[i])
        answer_file.write(ans_set[i])


    question_file.close()
    answer_file.close()



if __name__ == '__main__':
    get_train_set()
    # print(train_set)
    # print('no repeat num: ',len(train_set))
    # get_train_set('./samples/backup/question','./samples/backup/answer')
    # print(train_set)
    # print('no repeat num: ',len(train_set))
    save_train_set()
    print('save successfully......')
