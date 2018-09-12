# coding: utf-8

# In[12]:


import tensorflow as tf
import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', unpack=True, dtype='float32')
x_data = np.transpose(data[0:2])
y_data = np.transpose(data[2:])
global_step = tf.Variable(0, trainable=False, name='global_step')

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# In[13]:


# 신경망 계층에 레이어 추가
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name='W1')
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('layer3'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name='W3')
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost, global_step=global_step)

# 손실값 추적,
tf.summary.scalar('cost', cost)

# 각 가중치 및 편향등의 변화 확인
tf.summary.histogram("Weights", W1)

# In[14]:


sess = tf.Session()
# 앞에서 정의한 변수들을 가져와 파일에 저장하거나, 이전에 학습한 결과를 불러와 담는 변수 사용
saver = tf.train.Saver(tf.global_variables())

ckpt = tf.train.get_checkpoint_state('./model')
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    # 학습된 값 불러오기
    saver.restore(sess, ckpt.model_checkpoint_path)
else:
    # 변수 초기화
    sess.run(tf.global_variables_initializer())

# 앞서 지정된 텐서들 수집하고, 그래프와 텐서들의 값을 저장할 디렉토리 설정
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('./logs', sess.graph)

for step in range(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})
    print('Step: %d, ' % sess.run(global_step),
          'Cost: %.3f' % sess.run(cost, feed_dict={X: x_data, Y: y_data}))

    # sess.run을 사용하여 merged값들을 계산하여 수집한 후 저장, 나중에 확인할 수 있도록 global_step값 기록
    summary = sess.run(merged, feed_dict={X: x_data, Y: y_data})
    writer.add_summary(summary, global_step=sess.run(global_step))

# 최적화 후 학습된 변수들을 지정한 체크포인트파일 저장
saver.save(sess, './model/dnn.ckpt', global_step=global_step)


# In[ ]:


prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
print('실제값:', sess.run(target, feed_dict={Y: y_data}))
is_currect = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_currect, tf.float32))
print('정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
