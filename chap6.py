# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

# 표준편차 0.01
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
L1 = tf.nn.relu(tf.matmul(X, W1))
# 드롭아웃, (0.8 : 사용할 뉴런의 비율)
# 학습이 끝나고 예측시에는 전체를 사용해야 하므로 별도의 변수를 만들어 활용한다.
# L1 = tf.nn.dropout(L1, keep_prob)
# 배치정규화
L1 = tf.layers.batch_normalization(L1, training=True)


W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
L2 = tf.nn.relu(tf.matmul(L1, W2))
# L2 = tf.nn.dropout(L2, keep_prob)
# 배치정규화
L2 = tf.layers.batch_normalization(L2, training=True)

W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.nn.relu(tf.matmul(L2, W3))


# 각 이미지에 대한 손실값(실제값과 예측값 차이)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
# 손실값을 최소화하는 최적화
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# 세션 시작
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 미니배치 구함
batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

# 에포크 (학습데이터 전체를 한바퀴 도는것)
for epoch in range(30):
    total_cost = 0
    for i in range(total_batch):

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys, keep_prob: 0.8})

        total_cost += cost_val
    print('Epoch:', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료')

# 예측값(model) 중 가장 큰값이 가장 근접한 예측결과 (softmax_cross_entropy_with_logits 계산 결과)
# 1번 인덱스의 차원 값 중 최댓값의 인덱스 뽑아냄
is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))

# 정확도 계산
# is_correct를 0과 1로 변환
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

# 예측모델 실행후 결과값 저장
labels = sess.run(model,
                  feed_dict={X: mnist.test.images,
                             Y: mnist.test.labels,
                             keep_prob: 1})

fig = plt.figure()

for i in range(10):
    # 2행 5열 그래프 생성, i+1번째 숫자이미지 출력
    subplot = fig.add_subplot(2, 5, i + 1)
    # x, y축 눈금 제거
    subplot.set_xticks([])
    subplot.set_yticks([])

    # 출력한 이미지 위에 예측한 숫자 출력
    # 결과값인 labels의 해당 배열에서 가장 높은 값을 가진 인덱스를 예측한 숫자로 출력
    subplot.set_title('%d' % np.argmax(labels[i]))

    # 1차원 배열로 되어 있는 이미지 데이터를 28x28 형식의 2차원 배열로 변형하여 이미지 출력
    # cmap 파라메터를 통해 이미지를 그레이스케일로 출력
    subplot.imshow(mnist.test.images[i].reshape((28,28)), cmap=plt.cm.gray_r)

plt.show()