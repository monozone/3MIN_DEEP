# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

# 하이퍼파라메터 설정
total_epoch = 100
batch_size = 100
n_hidden = 256
n_input = 28 * 28
n_noise = 128
n_class = 10

X = tf.placeholder(tf.float32, [None, n_input])
# 노이즈와 실제값에 각각 해당 숫자를 힌트로 넣어주는 용도
Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, n_noise])


# 생성자 신경망
# tf.layers 사용
# GAN 모델은 생성자와 구분자를 동시에 학습시키고 각 신경망 변수들을 따로 학습시켜야 하지만
# tf.layers를 사용하면 변수를 선언하지 않고 tf.variable_scope를 이용하여 스코프에 해당되는 변수들만 따로 불러올 수 있음
# tf.concat 함수를 사용하여 noise값에 labels 정보를 추가
# tf.layers.dense 함수를 사용하여 은닉층 생성하고 출력층도 생성
def generator(noise, labels):
    with tf.variable_scope('generator'):
        inputs = tf.concat([noise, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, n_input, activation=tf.nn.sigmoid)
    return output


# 구분자 신경망
# 구분자는 진짜 이미지 판별할때와 가짜 이미지 판별할때 똑같은 변수를 사용해야 함
# 그러기 위해 scope.reuse_variables 함수를 이용해 이전에 사용한 변수를 재사용하도록 함.
def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        if reuse:
            scope.reuse_variables()
        inputs = tf.concat([inputs, labels], 1)
        hidden = tf.layers.dense(inputs, n_hidden, activation=tf.nn.relu)
        output = tf.layers.dense(hidden, 1, activation=None)
    return output

# 무작위 노이즈 생성 함수 (균등분포)
def get_noise(batch_size, noise):
    return np.random.uniform(-1., 1., size=[batch_size, noise])

# 레이블정보 추가하여 추후 레이블정보에 해당하는 이미지 생성 유도
G = generator(Z, Y)
D_real = discriminator(X, Y)
# 가짜 이미지 생성시 진짜 이미지에 사용되었던 변수들을 재사용하기 위해 reuse옵션을 True로 설정
D_gene = discriminator(G, Y, True)

# 손실값 구하기
# 가짜라고 판단하는 손실값 (D_real은 1에 가까워야 하고)
loss_D_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
# 진짜라고 판단하는 손실값 (D_gene는 0에 가까워야 한다)
loss_D_gene = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.zeros_like(D_gene)))

# 구분자 손실값, 둘을 합쳐서 최소화 하면 구분자(경찰)을 학습시킬 수 있음
loss_D = loss_D_real + loss_D_gene

# 생성자 손실값, ones_like함수를 사용하여 D_gene를 1에 가깝게 만듦
loss_G = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gene, labels=tf.ones_like(D_gene)))


# 학습모델 구성
# 각 스코프에서 사용된 변수들을 가져와 최적화에 사용될 각각의 손실함수와 함께 최적화 함수에 넣음
# 'discriminator' 스코프에서 사용된 변수 수집
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
# 최적화 함수에 넣음
train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list=vars_G)

# 학습
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# 미니배치
total_batch = int(mnist.train.num_examples / batch_size)
loss_val_D, loss_val_G = 0, 0

# 구분자와 생성자 신경망을 각각 학습 시킴
for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size, n_noise)

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Y: batch_ys, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Y: batch_ys, Z: noise})


    print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D),
                                    'G loss: {:.4}'.format(loss_val_G))

    # 확인용 이미지 생성
    if epoch == 0 or (epoch +1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        # 노이즈 생성 후 생성자 G에 넣어 결괏값 samples 생성
        samples = sess.run(G, feed_dict={Y: mnist.test.labels[:sample_size], Z: noise})

        fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

        for i in range(sample_size):
            ax[0][i].set_axis_off()
            ax[1][i].set_axis_off()
            ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
            ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('sample/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료')

