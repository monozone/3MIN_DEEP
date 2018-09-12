# coding: utf-8

import tensorflow as tf

hello = tf.constant('Hello Tensorflow!')

a = tf.constant(10)
b = tf.constant(32)
c = tf.add(a, b)
sess = tf.Session()
print(sess.run(hello))
print(sess.run([a, b, c]))
sess.close()

X = tf.placeholder(tf.float32, [None, 3])
x_data = [[1, 2, 3], [4, 5, 6]]
W = tf.Variable(tf.random_normal([3, 2]))  # [3,2] 텐서행렬
b = tf.Variable(tf.random_normal([2, 1]))  # [2,1] 텐서행렬
expr = tf.matmul(X, W) + b

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print("=== x_data ===")
print(x_data)
print("=== W ===")
print(sess.run(W))
print("=== b ===")
print(sess.run(b))
print("=== expr ===")
print(sess.run(expr, feed_dict={X: x_data}))
sess.close()


# 선형회귀 모델
x_data = [1, 2, 3]
y_data = [1, 2, 3]
W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # -1.0 ~ 1.0 까지 균등분포를 가진 무작위 값으로 초기화
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
hypothesis = W * X + b  # X가 주어졌을떄 Y를 만들어 낼 수 있는 W와 b를 찾는다. W는 가중치, b는 편향. 신경망 학습의 기본
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train_op = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(100):
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})
        print(step, cost_val, sess.run(W), sess.run(b))

    print("X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5}))
    print("X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5}))
