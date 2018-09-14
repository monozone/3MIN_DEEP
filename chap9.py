# -*- coding:utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('./mnist/data', one_hot=True)

# 하이퍼파라메터 설정
total_epoch = 1000
batch_size = 1000
learning_rate = 0.0002
n_hidden = 256
n_input = 28 * 28
n_noise = 128


X = tf.placeholder(tf.float32, [None, n_input])
Z = tf.placeholder(tf.float32, [None, n_noise])

# 생성자 신경망
# 은닉층으로 출력하기 위한 변수 (가중치, 편향)
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
# 출력층에 사용할 변수 (가중치, 편향), 가중치의 변수는 실제 이미지 크기와 같아야 함.
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 구분자 신경망
# 구분자는 진짜와 얼마나 가까운가를 판단하므로 0~1사이의 값을 출력
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


# 생성자 신경망 구성
# 무작위 생성한 노이즈를 받아 가중치와 편향을 반영하여 은닉층을 만들고 실제 이미지와 같은 크기의 갈괏값 출력
def generator(noise_z):
    # 가중치와 편향을 반영하여 은닉층을 만들고
    hidden = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    # 실제 이미지와 같은 크기의 갈괏값 출력
    output = tf.nn.sigmoid(tf.matmul(hidden, G_W2) + G_b2)
    return output

# 구분자 신경망 구성
def discriminator(inputs):
    hidden = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    output = tf.nn.sigmoid(tf.matmul(hidden, D_W2) + D_b2)
    return output

# 무작위 노이즈 생성 함수
def get_noise(batch_size, noise):
    return np.random.normal(size=(batch_size, noise))


# 가짜 이미지 생성하고 진짜와 가짜 구분
G = generator(Z)
D_gene = discriminator(G)
D_real = discriminator(X)

# 손실값 구하기
# 가짜라고 판단하는 손실값 (D_real은 1에 가까워야 하고)
# 진짜라고 판단하는 손실값 (D_gene는 0에 가까워야 한다)

# D_real과 1에서 각각 D_gene를 뺀값을 서로 더함 (경찰 학습)
loss_D = tf.reduce_mean(tf.log(D_real) +  tf.log(1 - D_gene))

# D_gene를 1에 가깝게 만들어야 함 (위조지폐범 학습), 가짜 이미지를 넣어도 진짜라고 해야 하기에
loss_G = tf.reduce_mean(tf.log(D_gene))

# GAN학습의 핵심은 loss_D와 loss_D 모두를 최대화 하는것

# 손실값을 이용해 학습
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

# 변수 최적화 : loss를 최대화해야 함.
# 최적화에 쓸수 있는 함수는 minimize밖에 없으므로 음수로 수행
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)

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

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})


    print('Epoch:', '%04d' % epoch, 'D loss: {:.4}'.format(loss_val_D),
                                    'G loss: {:.4}'.format(loss_val_G))

    # 확인용 이미지 생성

    if epoch == 0 or (epoch +1) % 10 == 0:
        sample_size = 10
        noise = get_noise(sample_size, n_noise)
        # 노이즈 생성 후 생성자 G에 넣어 결괏값 samples 생성
        samples = sess.run(G, feed_dict={Z: noise})

        fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))
        for i in range(sample_size):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))

        plt.savefig('sample/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)

print('최적화 완료')

