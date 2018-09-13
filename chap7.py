# -*- coding:utf-8 -*-

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)

# CNN은 2차원 평면이므로 직관적 형태로 구성됨.

# 입력데이터 갯수, 가로갯수, 세로갯수, 회색조색상이므로 색상한개
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
