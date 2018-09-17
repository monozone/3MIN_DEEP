# -*- coding:utf-8 -*-

import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data/', one_hot=True)


start = time.time()


learning_rate = 0.001               # 학습률
total_epoch = 30                    # 학습할 총 횟수
batch_size = 128                    # 미니배치로 한번에 학습할 데이터(이미지)의 갯수

n_input = 28                        # 이미지의 가로 한줄
n_step = 28                         # 이미지 한줄씩 세로 28번 순회
n_hidden = 128                      # 은닉층의 뉴런 갯수
n_class = 10

# 입력값
X = tf.placeholder(tf.float32, [None, n_step, n_input])
# 출력값
Y = tf.placeholder(tf.float32, [None, n_class])
# 가중치
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
# 편향
b = tf.Variable(tf.random_normal([n_class]))

# RNN셀 생성
cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

# RNN 신경망 완료
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# 결괏값을 원-핫 인코딩으로 구현, 손실함수로 tf.nn.softmax_cross_entropy_with_logits 사용
# tf.nn.softmax_cross_entropy_with_logits 는 최종결괏값이 실측값과 동일한 형태인 [batch_size, n_class]여야 함

# n_step과 batch_size 차원의 순서 바꿈
outputs = tf.transpose(outputs, [1, 0, 2])
# n_step차원 제거하고 마지막 결괏값만 수용
outputs = outputs[-1]

# 최종 결괏값 계산
model = tf.matmul(outputs, W) + b

# 실측값과 비교하여 손실값 구함
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# 최적화
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# 결과 확인
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 미니배치
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost =', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

end = time.time() - start
print(end)

test_batch_size = len(mnist.test.images)
test_xs = mnist.images.reshape(test_batch_size, n_step, n_input)
test_ys = mnist.test.labels

print('정확도:', sess.run(accuracy, feed_dict={X: test_xs, Y: test_ys}))


# 회사 구동 시간 :
# 173.0208613872528
# 175.8797698020935


# 집 구동 시간   :