




input_files = ['./data/test.answer.nosegment','./data/newseq2seq_generated_test.answer.nosegment','./data/oldseq2seq_generated_test.answer.nosegment']
lookup_files = ['./data/test_answer_vectors.dic','./data/newseq2seq_test_answer_vectors.dic','./data/oldseq2seq_test_answer_vectors.dic']
output_files = ['./data/test_answer_vectors.list','./data/newseq2seq_test_answer_vectors.list','./data/oldseq2seq_test_answer_vectors.list']

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

    input_file = open(lookup_files[i], "r")
    a = input_file.read()
    lookup_dic = eval(a)
    input_file.close()

    sentences_list = []
    count = 0
    for sentence in sentences:
        print('count............',count)
        count += 1
        single_list = []
        single_list.append(sentence)
        single_list.append(lookup_dic[sentence])
        sentences_list.append(single_list)

    f = open(output_files[i],'w')
    f.write(str(sentences_list))
    f.close()
    print('success......')
    print('input_file',input_files[i])
    print('lookup_file',lookup_files[i])
    print('output_file',output_files[i])
