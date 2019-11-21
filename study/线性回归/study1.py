#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 9:37 PM
# @Author  : w8ay
# @File    : study1.py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

points_num = 100
vectors = []

for i in range(points_num):
    x1 = np.random.normal(0.0, 0.66)
    y1 = 0.1 * x1 + 0.2 + np.random.normal(0.0, 0.04)
    vectors.append([x1, y1])

x_data = [i[0] for i in vectors]
y_data = [i[1] for i in vectors]

# 图像展示
# plt.plot(x_data,y_data,'r*',label="Origin Data")
# plt.legend()
# plt.show()

W = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

learn_rate = 0.5
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learn_rate)
train = optimizer.minimize(loss)

with tf.compat.v1.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 训练20步
    for i in range(20):
        sess.run(train)
        print("Step:{} Loss:{} W:{} b:{}".format(i, sess.run(loss), sess.run(W), sess.run(b)))

    plt.plot(x_data, y_data, 'r*', label="Origin Data")
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), label="line")
    plt.legend()
    plt.show()
