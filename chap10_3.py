# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np


char_arr = [c for c in "SEPabcdefghijklmnopqrstuvwxyz단어나무놀이소녀키스사랑"]
num_dic = {n: i for i, n in enumerate(char_arr)}
dic_len = len(num_dic)

seq_data = [['word', '단어'], ['wood', '나무'], ['game', '놀이'], ['girl', '소녀'], ['kiss', '키스'], ['love', '사랑']]


def make_batch(seq_data):
    input_batch = []
    output_batch = []
    target_batch = []

    for seq in seq_data:
        # 데이터는 인코더의 입력값, 디코더의 입력값, 출력값 총 3개로 구성
        # 인코더 셀의 입력값을 위해 입력단어를 한글자씩 떼어 배열로 만듦
        input = [num_dic[n] for n in seq[0]]
        # 디코더 셀의 입력값을 위해 출력단어의 글자들을 배열로 만들고 시작을 나타내는 심볼 'S'를 맨앞에 추가
        output = [num_dic[n] for n in ('S' + seq[1])]
        # 학습을 위해 비교할 디코더셀의 출력값을 만들고 출력 끝을 알려주는 심볼 'E'를 마지막에 붙입
        target = [num_dic[n] for n in (seq[1] + 'E')]

        # 단위행렬로 변환
        input_batch.append(np.eye(dic_len)[input])
        output_batch.append(np.eye(dic_len)[output])
        target_batch.append(target)

    return input_batch, output_batch, target_batch


# 하이퍼파라메터
learning_rate = 0.01
n_hidden = 128
total_epoch = 400
# 입출력에 사용할 글자들의 배열 크기 (사용할 전체 글자들)
n_class = n_input = dic_len

# 인코더,디코더 입력값
# [batch size, time steps, input size]
# 디코더 출력값
# [batch size, time steps]

####################################################################
# 신경망 모델 구성
# RNN 특성상 입력데이터에 단계가 있음.
# 입력값들은 원-핫 인코딩을 사용하고 디코더의 출력값은 인덱스숫자를 사용하기 때문에 입력값의 랭크(차원)가 하나 더 높음
# 각 입력단계는 배치크기처럼 입력받을떄마다 다를 수 있으므로 None
# 현재 샘플은 같은 크기의 단어를 테스트함.
enc_input = tf.placeholder(tf.float32, [None, None, n_input])
dec_input = tf.placeholder(tf.float32, [None, None, n_input])
targets = tf.placeholder(tf.int64, [None, None])

# RNN 모델을 위한 셀 구성
# 셀은 기본셀을 사용하고, 각 셀에 드롭아웃 적용
# 주의할점 : 디코더의 초기상태값(입력값이 아님)은 인코더의 최종 상태값
with tf.variable_scope('encode'):
    enc_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    enc_cell = tf.nn.rnn_cell.DropoutWrapper(enc_cell, output_keep_prob=0.5)
    outputs, enc_states = tf.nn.dynamic_rnn(enc_cell, enc_input, dtype=tf.float32)

with tf.variable_scope('decode'):
    dec_cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)
    dec_cell = tf.nn.rnn_cell.DropoutWrapper(dec_cell, output_keep_prob=0.5)
    outputs, dec_states = tf.nn.dynamic_rnn(dec_cell, dec_input, initial_state=enc_states, dtype=tf.float32)

# 출력층 만들고 손실함수, 최적화함수 구성
# 가중치와 편향 없음, 고수준의 API를 사용하면 텐서플로가 알아서 해줌.
model = tf.layers.dense(outputs, n_class, activation=None)
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model, labels=targets))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)


####################################################################
# 학습 시작
# 학습데이터는 인코더의 입력값, 디코더의 입력값과 출력값 3개로 구성
sess = tf.Session()
sess.run(tf.global_variables_initializer())

input_batch, output_batch, target_batch = make_batch(seq_data)

for epoch in range(total_epoch):
    _, loss = sess.run([optimizer, cost],
                       feed_dict={enc_input: input_batch,
                                  dec_input: output_batch,
                                  targets: target_batch})

    print('Epoch:', '%04d' % (epoch + 1),
          'cost = ', '{:.6f}'.format(loss))

print('최적화 완료')


####################################################################
# 테스트 (단어를 입력받아 번역단어 예측)
# 입력값과 출력값을 [영어, 한글] 사요
# 예측시에는 한글을 인식 못하므로 디코더의 입출력을 의미없는 값인 'P'로 채움

def translate(word):
    seq_data = [word, 'P' * len(word)]
    input_batch, output_batch, target_batch = make_batch(seq_data)
    # 입력값 = 'word',
    # input_batch = ['w','o','r','d']
    # output_batch = ['P','P','P','P']에 대한 원-핫 인코딩 값
    # target_batch=['P','P','P','P']에 대한 인덱스값인 [2,2,2,2]

    # 예측모델 동작
    # 결괏값이 [batch size, time steps, input size] 이므로 세번째 차원을 argmax로 취해 가장 확률이 높은 글자를 예측값으로 만듦
    prediction = tf.argmax(model, 2)
    result = sess.run(prediction, feed_dict={enc_input: input_batch,
                                             dec_input: output_batch,
                                             targets: target_batch})

    # 결괏값은 글자의 인덱스를 뜻하는 숫자이므로 각 숫자에 해당하는 글자를 가져오고
    # 출력의 끝을 의미하는 'E' 제거
    decoded = [char_arr[i] for i in result[0]]


    if 'E' in decoded:
        end = decoded.index('E')
    else:
        end = len(decoded)
    translated = ''.join(decoded[:end])
    return translated


####################################################################
# 번역 테스트
print('\n=== 번역 테스트 ===')
print('word => ', translate('word'))
print('wodr => ', translate('wodr'))
print('love => ', translate('love'))
print('loev => ', translate('loev'))
print('abcd => ', translate('abcd'))
