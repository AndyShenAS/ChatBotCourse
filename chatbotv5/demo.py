# coding:utf-8
# author: lichuang
# mail: shareditor.com@gmail.com
import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import word_token
import jieba
import random
import time


# 空值填充0
PAD_ID = 0
# 输出序列起始标记
GO_ID = 1
# 结尾标记
EOS_ID = 2
UNK_ID = 3


# 在样本中出现频率超过这个值才会进入词表

# tensorboard --logdir=/tmp/tensorflow/old_seq2seq_logs/ --port=6006
wordToken = word_token.WordToken()
# saver.save(sess, './model/bigcorpus/demo')
log_dir = '/tmp/tensorflow/old_seq2seq_logs/'
difstr = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
log_dir += difstr
option = 1

if option == 1:
    question_path = './samples/backup/question.segment'
    answer_path = './samples/backup/answer.segment'
    train_set_modify = 0
    model_path = './model/demo'
    batchNUM = 1000
    learning_rate_threshold = 5
    min_freq = 1

    # 输入序列长度
    input_seq_len = 7
    # 输出序列长度
    output_seq_len = 9
    # LSTM神经元size
    size = 10
    # 初始学习率
    init_learning_rate = 0.01
    Epoches = 500
    # step= 49990 loss= 0.57324296 learning_rate= 0.007855157
    # step= 49990 loss= 0.5078533 learning_rate= 0.004174552
    # step= 99990 loss= 0.45555034 learning_rate= 0.0017970073
    # step= 199990 loss= 0.4141012 learning_rate= 0.0010611147
    #step= 990 loss= 0.41393718 learning_rate= 0.0010611147


elif option == 2:
    # question_path = './samples/question.big'
    # answer_path = './samples/answer.big'
    question_path = './samples/question.big.norepeat.segment'
    answer_path = './samples/answer.big.norepeat.segment'
    train_set_modify = 0
    model_path = './model/bigcorpus/demo'
    batchNUM = 256
    learning_rate_threshold = 8
    min_freq = 40
    # 输入序列长度
    input_seq_len = 5
    # 输出序列长度
    output_seq_len = 7
    # LSTM神经元size
    size = 8
    # 初始学习率
    init_learning_rate = 0.01
    Epoches = 100


elif option == 3:
    question_path = './samples/backup/question'
    answer_path = './samples/backup/answer'
    model_path = './model/autoencoder/demo'
    train_set_modify = 1
    batchNUM = 256
    learning_rate_threshold = 5
    min_freq = 1
    # 输入序列长度
    input_seq_len = 7
    # 输出序列长度
    output_seq_len = 9
    # LSTM神经元size
    size = 10
    # 初始学习率
    # init_learning_rate = 1
    init_learning_rate = 0.0070696413
    Epoches = 50
    # step= 9990 loss= 0.55949193 learning_rate= 0.04239112
    # step= 49990 loss= 0.26494506 learning_rate= 0.009697726
    # step= 49990 loss= 0.18960254 learning_rate= 0.0070696413






# 放在全局的位置，为了动态算出num_encoder_symbols和num_decoder_symbols
max_token_id = wordToken.load_segment_file_list([question_path, answer_path], min_freq)
# max_token_id = wordToken.load_file_list(['./samples/question.big', './samples/answer.big'], min_freq)
num_encoder_symbols = max_token_id + 5
num_decoder_symbols = max_token_id + 5
print('num_symbols： ',num_decoder_symbols)


def get_id_list_from(sentence):
    sentence_id_list = []
    seg_list = jieba.cut(sentence)
    for str in seg_list:
        id = wordToken.word2id(str)
        if id:
            sentence_id_list.append(wordToken.word2id(str))
    return sentence_id_list

def get_id_list_from_seq(sentence_seq):
    sentence_id_list = []
    for str in sentence_seq:
        id = wordToken.word2id(str)
        if id:
            sentence_id_list.append(wordToken.word2id(str))
    return sentence_id_list


def get_train_set():
    global num_encoder_symbols, num_decoder_symbols
    train_set = []
    count = 0
    with open(question_path, 'r') as question_file:
        with open(answer_path, 'r') as answer_file:
            while True:
                count += 1
                question_seq = []
                answer_seq = []
                question = question_file.readline()
                answer = answer_file.readline()
                if question and answer:
                    line_question = question.strip()
                    line_answer = answer.strip()
                    if count<30:
                        print(line_question)
                        print(line_answer)
                    for word in line_question.split(' '):
                        question_seq.append(word)
                    for word in line_answer.split(' '):
                        answer_seq.append(word)

                    question_id_list = get_id_list_from_seq(question_seq)
                    answer_id_list = get_id_list_from_seq(answer_seq)
                    if len(question_id_list) > 0 and len(answer_id_list) > 0 and len(question_id_list) <= input_seq_len and len(answer_id_list) <= output_seq_len-2:
                        if len(question_seq) - len(question_id_list) < 2 and len(answer_seq) - len(answer_id_list) < 1:
                            if train_set_modify == 1:
                                print('yes')
                                question_id_list_noEOS = question_id_list[:]
                                question_id_list.append(EOS_ID)
                                train_set.append([question_id_list_noEOS, question_id_list])
                                answer_id_list_noEOS = answer_id_list[:]
                                answer_id_list.append(EOS_ID)
                                train_set.append([answer_id_list_noEOS, answer_id_list])
                            else:
                                answer_id_list.append(EOS_ID)
                                train_set.append([question_id_list, answer_id_list])

                else:
                    break

    print('tainset length: ',len(train_set))
    return train_set

def get_order_train_set(train_set):
    orderd_train_set = []
    for length in range(1,input_seq_len+1):
        for sample in train_set:
            if len(sample[0]) == length:
                orderd_train_set.append(sample)
    return orderd_train_set



def get_samples(train_set, batch_num):
    """构造样本数据

    :return:
        encoder_inputs: [array([0, 0], dtype=int32), array([0, 0], dtype=int32), array([5, 5], dtype=int32),
                        array([7, 7], dtype=int32), array([9, 9], dtype=int32)]
        decoder_inputs: [array([1, 1], dtype=int32), array([11, 11], dtype=int32), array([13, 13], dtype=int32),
                        array([15, 15], dtype=int32), array([2, 2], dtype=int32)]
    """
    # train_set = [[[5, 7, 9], [11, 13, 15, EOS_ID]], [[7, 9, 11], [13, 15, 17, EOS_ID]], [[15, 17, 19], [21, 23, 25, EOS_ID]]]
    for i in range(len(train_set)//batch_num + 1):
        raw_encoder_input = []
        raw_decoder_input = []
        # print('length of trainset: ',len(train_set))
        # batch_train_set = train_set
        if batch_num >= len(train_set):
            batch_train_set = train_set
        else:
            # random_start = random.randint(0, len(train_set)-batch_num)
            # batch_train_set = train_set[random_start:random_start+batch_num]
            start_i = i * batch_num
            end_i = start_i + batch_num
            if end_i > len(train_set) - 1:
                end_i = len(train_set) - 1
                start_i = end_i - batch_num
            batch_train_set = train_set[start_i:end_i]
        for sample in batch_train_set:
            raw_encoder_input.append([PAD_ID] * (input_seq_len - len(sample[0])) + list(reversed(sample[0])))
            raw_decoder_input.append([GO_ID] + sample[1] + [PAD_ID] * (output_seq_len - len(sample[1]) - 1))

        encoder_inputs = []
        decoder_inputs = []
        target_weights = []

        for length_idx in range(input_seq_len):
            encoder_inputs.append(np.array([encoder_input[length_idx] for encoder_input in raw_encoder_input], dtype=np.int32))
        for length_idx in range(output_seq_len):
            decoder_inputs.append(np.array([decoder_input[length_idx] for decoder_input in raw_decoder_input], dtype=np.int32))
            target_weights.append(np.array([
                0.0 if length_idx == output_seq_len - 1 or decoder_input[length_idx] == PAD_ID else 1.0 for decoder_input in raw_decoder_input
            ], dtype=np.float32))
        # return encoder_inputs, decoder_inputs, target_weights
        yield encoder_inputs, decoder_inputs, target_weights


def seq_to_encoder(input_seq):
    """从输入空格分隔的数字id串，转成预测用的encoder、decoder、target_weight等
    """
    input_seq_array = [int(v) for v in input_seq.split()]
    encoder_input = [PAD_ID] * (input_seq_len - len(input_seq_array)) + input_seq_array
    decoder_input = [GO_ID] + [PAD_ID] * (output_seq_len - 1)
    encoder_inputs = [np.array([v], dtype=np.int32) for v in encoder_input]
    decoder_inputs = [np.array([v], dtype=np.int32) for v in decoder_input]
    target_weights = [np.array([1.0], dtype=np.float32)] * output_seq_len
    return encoder_inputs, decoder_inputs, target_weights

model_gradient_clipping = 1
clip_value_min = -3
clip_value_max = 3
clip_norm = 5


def get_model(feed_previous=False):
    """构造模型
    """
    global learning_rate

    learning_rate = tf.Variable(float(init_learning_rate), trainable=False, dtype=tf.float32)
    learning_rate_decay_op = learning_rate.assign(learning_rate * 0.9)
    learning_rate_increase_op = learning_rate.assign(learning_rate * 1.1)

    encoder_inputs = []
    decoder_inputs = []
    target_weights = []
    for i in range(input_seq_len):
        encoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))
    for i in range(output_seq_len + 1):
        decoder_inputs.append(tf.placeholder(tf.int32, shape=[None], name="decoder{0}".format(i)))
    for i in range(output_seq_len):
        target_weights.append(tf.placeholder(tf.float32, shape=[None], name="weight{0}".format(i)))

    # decoder_inputs左移一个时序作为targets
    targets = [decoder_inputs[i + 1] for i in range(output_seq_len)]

    cell = tf.contrib.rnn.BasicLSTMCell(size)

    # 这里输出的状态我们不需要
    outputs, _ = seq2seq.embedding_attention_seq2seq(
                        encoder_inputs,
                        decoder_inputs[:output_seq_len],
                        cell,
                        num_encoder_symbols=num_encoder_symbols,
                        num_decoder_symbols=num_decoder_symbols,
                        embedding_size=size,
                        output_projection=None,
                        feed_previous=feed_previous,
                        dtype=tf.float32)

    # 计算加权交叉熵损失
    loss = seq2seq.sequence_loss(outputs, targets, target_weights)
    ######################
    tf.summary.scalar('loss',loss)
    ###########################
    # # 梯度下降优化器
    # opt = tf.train.GradientDescentOptimizer(learning_rate)
    # # 优化目标：让loss最小化
    # update = opt.apply_gradients(opt.compute_gradients(loss))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # Gradient Clipping
    gradients = optimizer.compute_gradients(loss)
    if model_gradient_clipping == 1:
        capped_gradients = [(tf.clip_by_value(grad, clip_value_min, clip_value_max), var) for grad, var in gradients if grad is not None]
    else:
        capped_gradients = [(tf.clip_by_norm(grad, clip_norm), var) for grad, var in grfadients if grad is not None]
    update = optimizer.apply_gradients(capped_gradients)


    # 模型持久化
    saver = tf.train.Saver(tf.global_variables())

    return encoder_inputs, decoder_inputs, target_weights, outputs,loss, update, saver, learning_rate_decay_op, learning_rate_increase_op, learning_rate


def train():
    """
    训练过程
    """
    # train_set = [[[5, 7, 9], [11, 13, 15, EOS_ID]], [[7, 9, 11], [13, 15, 17, EOS_ID]],
    #              [[15, 17, 19], [21, 23, 25, EOS_ID]]]
    train_set = get_train_set()
    train_set = get_order_train_set(train_set)
    print('load trainset successfully.....')
    print('length of trainset: ',len(train_set))
    with tf.Session() as sess:

        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate_increase_op, learning_rate = get_model()

        # 全部变量初始化
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, model_path)   #换这句可以接着上次的训练
        print('get model successfully.....')

        ####################################################
        merged_summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
        ####################################################

        # 训练很多次迭代，每隔10次打印一次loss，可以看情况直接ctrl+c停止
        previous_losses = []
        for step in range(Epoches):
            Epoches_losses = []
            for batch_i, (sample_encoder_inputs, sample_decoder_inputs, sample_target_weights) in enumerate(get_samples(train_set, batchNUM)):
                input_feed = {}
                for l in range(input_seq_len):
                    input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(output_seq_len):
                    input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[target_weights[l].name] = sample_target_weights[l]
                input_feed[decoder_inputs[output_seq_len].name] = np.zeros([len(sample_decoder_inputs[0])], dtype=np.int32)
                [loss_ret, _, summary_str] = sess.run([loss, update, merged_summary_op], input_feed)

                ###################################################
                summary_writer.add_summary(summary_str, step*(len(train_set)//batchNUM + 1) + batch_i)
                ###################################################
                Epoches_losses.append(loss_ret)

                if batch_i % 5 == 0:
                    print('step=', step,'batch_i=', batch_i, 'loss=', loss_ret, 'learning_rate=', learning_rate.eval())

            if len(previous_losses) > learning_rate_threshold and loss_ret > max(previous_losses[-learning_rate_threshold:]):
                sess.run(learning_rate_decay_op)
                print('update.......................... learning_rate=', learning_rate.eval())
            # if len(previous_losses) > learning_rate_threshold and loss_ret < min(previous_losses[-learning_rate_threshold:]):
            #     sess.run(learning_rate_increase_op)
            previous_losses.append(sum(Epoches_losses)/len(Epoches_losses))


            # 模型持久化 , 最后再保存
            saver.save(sess, model_path)




def predict():
    """
    预测过程
    """
    with tf.Session() as sess:
        encoder_inputs, decoder_inputs, target_weights, outputs, loss, update, saver, learning_rate_decay_op, learning_rate_increase_op, learning_rate = get_model(feed_previous=True)
        saver.restore(sess, model_path)
        sys.stdout.write("> ")
        sys.stdout.flush()
        input_seq = sys.stdin.readline()
        while input_seq:
            input_seq = input_seq.strip()
            input_id_list = get_id_list_from(input_seq)
            if (len(input_id_list)):
                sample_encoder_inputs, sample_decoder_inputs, sample_target_weights = seq_to_encoder(' '.join([str(v) for v in input_id_list]))

                input_feed = {}
                for l in range(input_seq_len):
                    input_feed[encoder_inputs[l].name] = sample_encoder_inputs[l]
                for l in range(output_seq_len):
                    input_feed[decoder_inputs[l].name] = sample_decoder_inputs[l]
                    input_feed[target_weights[l].name] = sample_target_weights[l]
                input_feed[decoder_inputs[output_seq_len].name] = np.zeros([2], dtype=np.int32)

                # 预测输出
                outputs_seq = sess.run(outputs, input_feed)
                # 因为输出数据每一个是num_decoder_symbols维的，因此找到数值最大的那个就是预测的id，就是这里的argmax函数的功能
                outputs_seq = [int(np.argmax(logit[0], axis=0)) for logit in outputs_seq]
                # 如果是结尾符，那么后面的语句就不输出了
                if EOS_ID in outputs_seq:
                    outputs_seq = outputs_seq[:outputs_seq.index(EOS_ID)]
                outputs_seq = [wordToken.id2word(v) for v in outputs_seq]
                print(" ".join(outputs_seq))
            else:
                print("WARN：词汇不在服务区")

            sys.stdout.write("> ")
            sys.stdout.flush()
            input_seq = sys.stdin.readline()


if __name__ == "__main__":
    if sys.argv[1] == 'train':
        train()
    else:
        predict()
