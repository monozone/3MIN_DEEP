# -*- coding:utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)


# 하이퍼파라메터
learning_rate = 0.1             # 학습률
training_epoch = 20             # 학습할 총 횟수
batch_size = 100                # 미니배치로 한번에 학습할 데이터(이미지)의 갯수
n_hidden = 256                  # 은닉층의 뉴런 갯수
n_input = 28 * 28               # 이미지의 가로세로 크기 = 784

# 비지도학습이므로 Y값이 없음
X = tf.placeholder(tf.float32, [None, n_input])

# 인코더, 디코더 생성 (핵심)
# 입력값(n_input)보다 은닉층(n_hidden)이 더 크다!!!
# n_hidden개의 뉴런을 가진 은닉층 생성
# 가중치와 편향변수를 뉴런의 갯수만큼 설정하고 입력값과 곱하고 더함
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
# 활성화 함수 적용
encoder = tf.nn.sigmoid(tf.add(tf.matmul(X, W_encode), b_encode))

# 디코더 생성
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))
decoder = tf.nn.sigmoid(tf.add(tf.matmul(encoder, W_decode), b_decode))

# 손실함수
# 실측값(X)과 디코더의 결과값(decoder)의 차이를 거리값(pow)으로 설정
cost = tf.reduce_mean(tf.pow(X - decoder, 2))

# 최적화
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


# 결과 확인
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(training_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        total_cost += cost_val

    print('Epoch: ', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료')

sample_size = 10
# 결과값 이미지 생성
samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

# 입력값과 신경망이 생성한 이미지 출력
fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()