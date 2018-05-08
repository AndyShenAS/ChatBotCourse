# coding:utf-8
import sys
import jieba


question_path = './samples/question.big'
answer_path = './samples/answer.big'

question_path_out = '../chatbotv2/data/question.all.norepeat'
answer_path_out = '../chatbotv2/data/answer.all.norepeat'




train_set = {}

def get_train_set(question_path = question_path ,answer_path = answer_path):
    count = 0
    with open(question_path, 'r') as question_file:
        with open(answer_path, 'r') as answer_file:
            while True:
                question = question_file.readline()
                answer = answer_file.readline()
                count += 1
                if question and answer:
                    question = question.strip()
                    answer = answer.strip()
                    if question in train_set:
                        train_set[question].append(answer)
                    else:
                        train_set[question] = [answer]
                else:
                    break
    print('original num: ', count)

def save_train_set(train_set):
    question_file = open(question_path_out, 'w')
    answer_file = open(answer_path_out, 'w')
    for key in train_set:
        value = train_set[key]
        question_file.write(key+'\n')
        answer_file.write(value[0]+'\n')

    question_file.close()
    answer_file.close()



if __name__ == '__main__':
    get_train_set()
    # print(train_set)
    print('no repeat num: ',len(train_set))
    get_train_set('./samples/backup/question','./samples/backup/answer')
    # print(train_set)
    print('no repeat num: ',len(train_set))
    save_train_set(train_set)
    print('save successfully......')
