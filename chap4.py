# coding: utf-8

# In[60]:


import tensorflow as tf
import numpy as np

# [털, 날개]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]]
)

# 기타 [1, 0, 0]
# 포유류 [0, 1, 0]
# 조류 [0, 0, 1]

y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

# 신경망 모델 구성

# X,Y 실측값 학습 설정
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 가중치 = [입력층(특징수), 출력층(레이블 수)] = 2, 3
# W = tf.Variable(tf.random_uniform([2, 3], -1.0, 1.0))

# 심층 신경망
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

# 편향 = 레이블수 3개
# b = tf.Variable(tf.zeros([3]))

# 심층 신경망
b1 = tf.Variable(tf.zeros([10]))  # 은닉층 뉴런 수
b2 = tf.Variable(tf.zeros([3]))

# ReLU 구성
# L = tf.add(tf.matmul(X, W), b)
# L = tf.nn.relu(L)

# 심층 신경망
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)

# 배열내 결과값을 전체합이 1이 되도록 수정 (확률 해석 위해)
# model = tf.nn.softmax(L)

# 심층 신경망 : 두번째 가중치와 편향 적용
model = tf.add(tf.matmul(L1, W2), b2)

# In[61]:


# 손실함수 기본 코드 (교체 엔트로피)
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

# 심층신경망
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))

# 기본적인 경사하강법으로 최적화
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

# 좀더 좋은 최적화
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

# 텐서플로 세션 초기화
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))

# In[62]:


# 학습결과 확인
prediction = tf.argmax(model, axis=1)  # 예측
target = tf.argmax(Y, axis=1)  # 결과
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))

# 정확도 출력
is_currect = tf.equal(prediction, target)
# print(sess.run(is_currect, feed_dict={X: x_data, Y: y_data}))
accuracy = tf.reduce_mean(tf.cast(is_currect, tf.float32))
# print(sess.run(accuracy, feed_dict={X: x_data, Y: y_data}))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
