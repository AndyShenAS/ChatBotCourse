import pandas as pd

input_files = ['./data/test.question.nosegment','./data/test.answer.nosegment','./data/newseq2seq_generated_test.answer.nosegment','./data/oldseq2seq_generated_test.answer.nosegment']

files_sentences = []
for i in range(len(input_files)):
    sentences = []
    with open(input_files[i], 'r') as input_file:
        while True:
            question = input_file.readline()
            if question:
                line_question = question.strip()
                sentences.append(line_question)
            else:
                break
    print('length of sentences {}'.format(i),len(sentences))
    files_sentences.append(sentences)

save_sentences = []
for j in range(len(files_sentences[0])):
    sentences = []
    for i in range(len(files_sentences)):
        sentences.append(files_sentences[i][j])
    save_sentences.append(sentences)





names = ['test.question','test.answer','newseq2seq_generated_test.answer','oldseq2seq_generated_test.answer']

file_saver = pd.DataFrame(columns = names, data = save_sentences)
file_saver.to_csv('./data/compare_answers.csv')
