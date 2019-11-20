#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 2:01 PM
# @Author  : w8ay
# @File    : framework.py
from conf.trainer_conf import RESIZE, IMAGE_CHANNEL, NUM_HIDDEN, NUM_CLASSES, CTC_MERGE_REPEATED, \
    PREPROCESS_COLLAPSE_REPEATED, CTC_LOSS_TIME_MAJOR, TRAINS_LEARNING_RATE, DECAY_STEPS, DECAY_RATE, NEU_OPTIMIZER, \
    MOMENTUM, CTC_BEAM_WIDTH, CTC_TOP_PATHS, RunMode
from lib.constants import OPTIMIZER_MAP, Optimizer
from lib.network.CNN import CNN5
from lib.network.LSTM import LSTM
from lib.optimizer.AdaBound import AdaBoundOptimizer
from lib.utils import NetworkUtils
import tensorflow as tf


class GraphOCR(object):

    def __init__(self, mode):
        self.utils = NetworkUtils(mode)
        self.inputs = tf.placeholder(tf.float32, [None, None, RESIZE[1], IMAGE_CHANNEL], name='input')
        self.labels = tf.sparse_placeholder(tf.int32, name='labels')
        self.seq_len = None
        self.merged_summary = None

    def build_graph(self):
        self._build_model()
        self._build_train_op()
        self.merged_summary = tf.summary.merge_all()

    def _build_model(self):
        # CNN 网络
        x = CNN5(inputs=self.inputs, utils=self.utils).build()
        # x = ResNet50(inputs=self.inputs, utils=self.utils).build()

        print("CNN Output: {}".format(x.get_shape()))
        self.seq_len = tf.fill([tf.shape(x)[0]], tf.shape(x)[1], name="seq_len")

        recurrent_network_builder = LSTM(self.utils, x, self.seq_len)
        # recurrent_network_builder = BLSTM(self.utils, x, self.seq_len)

        outputs = recurrent_network_builder.build()

        # Reshaping to apply the same weights over the time_steps
        outputs = tf.reshape(outputs, [-1, NUM_HIDDEN * 2])
        with tf.variable_scope('output'):
            # tf.Variable
            weight_out = tf.get_variable(
                name='weight',
                # outputs.get_shape()[1] if self.network == CNNNetwork.ResNet
                shape=[NUM_HIDDEN * 2, NUM_CLASSES],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer(),
            )
            biases_out = tf.get_variable(
                name='biases',
                shape=[NUM_CLASSES],
                dtype=tf.float32,
                initializer=tf.constant_initializer(value=0, dtype=tf.float32)
            )
            # [batch_size * max_timesteps, num_classes]
            logits = tf.matmul(outputs, weight_out) + biases_out
            # Reshaping back to the original shape
            logits = tf.reshape(logits, [tf.shape(x)[0], -1, NUM_CLASSES])
            # Time major
            predict = tf.transpose(logits, (1, 0, 2), "predict")
            self.predict = predict

    def _build_train_op(self):
        self.global_step = tf.train.get_or_create_global_step()
        # ctc loss function, using forward and backward algorithms and maximum likelihood.
        self.loss = tf.nn.ctc_loss(
            labels=self.labels,
            inputs=self.predict,
            sequence_length=self.seq_len,
            ctc_merge_repeated=CTC_MERGE_REPEATED,
            preprocess_collapse_repeated=PREPROCESS_COLLAPSE_REPEATED,
            ignore_longer_outputs_than_inputs=False,
            time_major=CTC_LOSS_TIME_MAJOR
        )

        self.cost = tf.reduce_mean(self.loss)
        tf.summary.scalar('cost', self.cost)
        self.lrn_rate = tf.train.exponential_decay(
            TRAINS_LEARNING_RATE,
            self.global_step,
            DECAY_STEPS,
            DECAY_RATE,
            staircase=True
        )
        tf.summary.scalar('learning_rate', self.lrn_rate)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # print(update_ops)
        # Storing adjusted smoothed mean and smoothed variance operations
        with tf.control_dependencies(update_ops):
            if OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.AdaBound:
                self.train_op = AdaBoundOptimizer(
                    learning_rate=self.lrn_rate,
                    final_lr=0.1,
                    beta1=0.9,
                    beta2=0.999,
                    amsbound=True
                ).minimize(
                    loss=self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.Adam:
                self.train_op = tf.train.AdamOptimizer(
                    learning_rate=self.lrn_rate
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.Momentum:
                self.train_op = tf.train.MomentumOptimizer(
                    learning_rate=self.lrn_rate,
                    use_nesterov=True,
                    momentum=MOMENTUM,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.SGD:
                self.train_op = tf.train.GradientDescentOptimizer(
                    learning_rate=self.lrn_rate,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.AdaGrad:
                self.train_op = tf.train.AdagradOptimizer(
                    learning_rate=self.lrn_rate,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )
            elif OPTIMIZER_MAP[NEU_OPTIMIZER] == Optimizer.RMSProp:
                self.train_op = tf.train.RMSPropOptimizer(
                    learning_rate=self.lrn_rate,
                    decay=DECAY_RATE,
                ).minimize(
                    self.cost,
                    global_step=self.global_step
                )

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        # self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(
        #     self.predict,
        #     self.seq_len,
        #     merge_repeated=False
        # )

        # Find the optimal path
        self.decoded, self.log_prob = tf.nn.ctc_beam_search_decoder(
            inputs=self.predict,
            sequence_length=self.seq_len,
            merge_repeated=False,
            beam_width=CTC_BEAM_WIDTH,
            top_paths=CTC_TOP_PATHS,
        )

        self.dense_decoded = tf.sparse.to_dense(self.decoded[0], default_value=-1, name="dense_decoded")


if __name__ == '__main__':
    GraphOCR(RunMode.Predict).build_graph()
