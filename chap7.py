# -*- coding:utf-8 -*-

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# CNN은 2차원 평면이므로 직관적 형태로 구성됨.

# 입력데이터 갯수, 가로갯수, 세로갯수, 회색조색상이므로 색상 한개
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# CNN계층 (컨볼루션) 구성
# 3x3크기의 커널을 오른쪽과 아래쪽으로 1칸씩 움직이는 32개의 커널을 가진 컨볼루션 계층 생성
# W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# 입력층 X와 가중치 W1, padding='SAME' : 커널 슬라이딩 시 이미지 가장 외곽에서 한칸 밖으로 움직이는 옵션 = 테두리까지 좀더 정확히 평가
# L1 = tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME')
# 활성화 함수로 컨볼루션 계층 생성
# L1 = tf.nn.relu(L1)
# CNN 첫번째 계층 (폴링) 구성
# 커널크기 2x2, strides = 슬라이딩시 두칸씩 움직임
# L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Layers 모듈 사용
L1 = tf.layers.conv2d(X, 32, [2, 2])
L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])



# CNN 2번째 계층
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 추출한 특징을 이용해 10개의 분류 생성
# 마지막 풀링크기는 7 * 7 * 64개, 중간단계 256개의 뉴런으로 연결
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
# 1차원 계층으로 변환
L3 = tf.reshape(L2, [-1, 7 * 7 * 64])
L3 = tf.matmul(L3, W3)
L3 = tf.nn.relu(L3)
# 과적합 막기
L3 = tf.nn.dropout(L3, keep_prob)

# Layers 모듈 사용
L3 = tf.contrib.layers.flatten(L2)
L3 = tf.layers.dense(L3, 256, activation=tf.nn.relu)



# L3의 출력값 256개를 받아 최종 출력값인 0~9레이블로 구성된 출력값 생성
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
# 텐서플로 계산
model = tf.matmul(L3, W4)

# 손실함수와 옵티마이저를 이용한 최적화
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)


# 결과 확인
init = tf.global_variables_initializer();
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples / batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)

        _, cost_val = sess.run([optimizer, cost],
                               feed_dict={X: batch_xs,
                                          Y: batch_ys,
                                          keep_prob: 0.7})
        total_cost += cost_val

    print('Epoch: ', '%04d' % (epoch + 1),
          'Avg. cost = ', '{:.3f}'.format(total_cost / total_batch))

print('최적화 완료')

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
                                  Y: mnist.test.labels,
                                  keep_prob: 1}))