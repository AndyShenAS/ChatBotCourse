
# f = open('./data/answer.segment')
# txt = f.read()
# print(txt)
# f.close()

wechat = ''
with open('./data/question.segment', 'r') as question_file:
    with open('./data/answer.segment', 'r') as answer_file:
        while True:
            question = question_file.readline()
            answer = answer_file.readline()
            if question and answer:
                wechat += question
                wechat += answer
            else:
                break

f = open('./data/corpus.segment.big','r')
txt = f.read()
for i in range(10):
    txt += wechat
f.close()


output_file = open('./data/wechat_corpus.segment.big', "w")
output_file.write(txt)
output_file.close()


# ../word2vec/word2vec -train ./data/corpus.segment.big -output ./data/wechat_vectors.bin -cbow 1 -size 200 -window 8 -negative 25 -hs 0 -sample 1e-5 -threads 20 -binary 1 -iter 15
