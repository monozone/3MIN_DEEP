# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np

char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
            'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# {'a': 0, 'b': 1, 'c': 2 ...}
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

# 학습단어
seq_data = ['word', 'wood', 'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']


# 단어를 학습형식으로 변환하는 함수
def make_batch(seq_data):
    input_batch = []
    target_batch = []

    for seq in seq_data:
        # 입력값용으로 단어의 처음 세글자 알파벳 인덱스 배열 생성
        input = [num_dic[n] for n in seq[:-1]]
        # 출력값으로 마지막 글자의 알파벳 인덱스 구함
        target = num_dic[seq[-1]]

        # 입력값을 원핫인코딩으로 변환
        input_batch.append(np.eye(dic_len)[input])
        target_batch.append(target)

    return input_batch, target_batch


learning_rate = 0.01               # 학습률
n_hidden = 128
total_epoch = 30                   # 학습할 총 횟수
n_step = 3                         # 단어 전체중 처음 3글자만 학습
n_input = n_class = dic_len

X = tf.placeholder(tf.float32, [None, n_step, n_input])
# 실측값으로 인덱스 숫자를 그대로 사용하므로 1차원 배열
Y = tf.placeholder(tf.int32, [None])
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

# 두개의 RNN셀 생성
cell1 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)
# 과적합방지를 위한 드롭아웃기법 사용
cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
cell2 = tf.nn.rnn_cell.BasicLSTMCell(n_hidden)

# MultiRNNCell을 사용하여 조합,
multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
# dynamic_rnn함수를 사요하여 심층순환신경망 구성
outputs, states = tf.nn.dynamic_rnn(multi_cell, X, dtype=tf.float32)

# 최종 출력층 생성
outputs = tf.transpose(outputs, [1, 0, 2])
outputs = outputs[-1]
model = tf.matmul(outputs, W) + b

# 손실화 및 최적화
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 신경망 학습
# make_batch함수를 이용하여 seq_data에 저장된 단어들의 입력값(처음3글자)과 실측값(마지막 한글자)로 분리하고
# 이 값들을 최적화 함수 실행코드에 넣어 신경망 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost], feed_dict={X: input_batch, Y: target_batch})
    print('Epoch:', '%04d' % (epoch + 1), 'cost = ', '{:.6f}'.format(loss))

print('최적화 완료!!')

# 정확도
prediction = tf.cast(tf.argmax(model, 1), tf.int32)
prediction_check = tf.equal(prediction, Y)
accuracy = tf.reduce_mean(tf.cast(prediction_check,  tf.float32))

# 학습에 사용한 단어로 예측모델 돌림
input_batch, target_batch = make_batch(seq_data)
predict, accuracy_val = sess.run([prediction, accuracy], feed_dict={X: input_batch, Y: target_batch})

# 모델이 예측한 값을 통해 각각 값에 해당하는 인덱스의 알파벳을 가져와 출력
predict_words = []
for idx, val in enumerate(seq_data):
    last_char = char_arr[predict[idx]]
    predict_words.append(val[:3] + last_char)

print('\n=== 예측 결과 ===')
print('입력값:', [w[:3] + ' ' for w in seq_data])
print('예측값:', predict_words)
print('정확도:', accuracy_val)

