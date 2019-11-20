#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 1:42 PM
# @Author  : w8ay
# @File    : trainer.py
import os
import numpy as np
import tensorflow as tf
from PIL import Image

from conf.trainer_conf import train_image_dir, test_image_dir, train_batch_size, test_batch_size, model_save_dir, \
    acc_stop, cycle_stop, cycle_save
from lib import framework, fileread
from lib.constants import RunMode
from tensorflow.python.framework.graph_util import convert_variables_to_constants


class Trainer():

    def __init__(self):
        print('Loading Trains DataSet...')
        self.trains_list = [os.path.join(os.path.realpath(train_image_dir), trains) for trains in
                            os.listdir(train_image_dir)]
        self.test_list = [os.path.join(os.path.realpath(test_image_dir), trains) for trains in
                          os.listdir(test_image_dir)]
        num_train_samples = len(self.trains_list)
        num_test_samples = len(self.test_list)

        np.random.shuffle(self.trains_list)
        np.random.shuffle(self.test_list)

        if num_train_samples < train_batch_size:
            raise Exception("训练批次要小于样本数量")
        if num_test_samples < test_batch_size:
            raise Exception("测试批次要小于样本数量")

    def train_process(self, mode=RunMode.Trains):
        model = framework.GraphOCR(mode)
        model.build_graph()

        print('Loading Test DataSet...')
        train_feeder = fileread.DataIterator(mode=RunMode.Trains)
        train_feeder.read_sample_from_files(self.trains_list)
        test_feeder = fileread.DataIterator(mode=RunMode.Test)
        test_feeder.read_sample_from_files(self.test_list)

        print('Total {} Trains DataSets'.format(train_feeder.size))
        print('Total {} Test DataSets'.format(test_feeder.size))

        num_batches_per_epoch = int(train_feeder.size / train_batch_size)
        accuracy = 0
        epoch_count = 1

        # 模型保存对象
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # 恢复模型
            if os.path.exists(model_save_dir):
                try:
                    saver.restore(sess, model_save_dir)
                    print("恢复模型成功")
                except ValueError:
                    print("model文件夹为空，将创建新模型")
            else:
                print("创建model文件夹")
                os.mkdir(model_save_dir)
            # 写入日志
            train_writer = tf.summary.FileWriter("logs/", sess.graph)
            print('Start training...')

            while 1:
                shuffle_trains_idx = np.random.permutation(train_feeder.size)
                last_train_avg_cost = 0
                for cur_batch in range(num_batches_per_epoch):
                    index_list = [
                        shuffle_trains_idx[i % train_feeder.size] for i in
                        range(cur_batch * train_batch_size, (cur_batch + 1) * train_batch_size)
                    ]
                    classified_batch = train_feeder.generate_batch_by_files(index_list)
                    step = 0
                    class_num = len(classified_batch)
                    avg_cost = 0

                    for index, (shape, batch) in enumerate(classified_batch.items()):
                        batch_inputs, batch_seq_len, batch_labels = batch
                        feed = {
                            model.inputs: batch_inputs,
                            model.labels: batch_labels,
                        }

                        summary_str, batch_cost, step, _ = sess.run(
                            [model.merged_summary, model.cost, model.global_step, model.train_op],
                            feed_dict=feed
                        )
                        avg_cost += batch_cost
                        last_train_avg_cost = avg_cost / class_num
                        train_writer.add_summary(summary_str, step)
                        if step % 10 == 0:
                            print('Step: {} , Cost = {:.5f} '.format(
                                step,
                                batch_cost,
                            ))
                        if step % cycle_save == 0:
                            saver.save(sess, model_save_dir)
                            print('Step: {0} 保存模型成功'.format(step))

                    if step % 500 == 0:
                        shuffle_test_idx = np.random.permutation(test_feeder.size)
                        index_test = [
                            shuffle_test_idx[i % test_feeder.size] for i in
                            range(cur_batch * test_batch_size, (cur_batch + 1) * test_batch_size)
                        ]
                        classified_batch = test_feeder.generate_batch_by_files(index_test)

                        all_dense_decoded = []
                        lr = 0

                        for index, (shape, batch) in enumerate(classified_batch.items()):
                            test_inputs, batch_seq_len, test_labels = batch
                            val_feed = {
                                model.inputs: test_inputs,
                                model.labels: test_labels
                            }
                            dense_decoded, sub_lr = sess.run(
                                [model.dense_decoded, model.lrn_rate],
                                feed_dict=val_feed
                            )
                            all_dense_decoded += dense_decoded.tolist()
                            lr += sub_lr
                        accuracy = fileread.accuracy_calculation(
                            test_feeder.labels,
                            all_dense_decoded,
                            ignore_value=[0, -1],
                        )
                        log = "Epoch: {}, Step: {}, accuracy = {:.5f}, " \
                              "LearningRate: {}"
                        print(log.format(
                            epoch_count,
                            step,
                            accuracy,
                            last_train_avg_cost, lr / len(classified_batch)
                        ))

                        if accuracy >= acc_stop and epoch_count >= cycle_stop:
                            break
                if accuracy >= acc_stop and epoch_count >= cycle_stop:
                    saver.save(sess, model_save_dir)
                    compile_graph(accuracy)
                    break

                epoch_count += 1


def compile_graph(acc):
    input_graph = tf.Graph()
    sess = tf.Session(graph=input_graph)

    with sess.graph.as_default():
        model = framework.GraphOCR(
            RunMode.Predict
        )
        model.build_graph()
        input_graph_def = sess.graph.as_graph_def()
        saver = tf.train.Saver(var_list=tf.global_variables())
        print(tf.train.latest_checkpoint(model_save_dir))
        saver.restore(sess, tf.train.latest_checkpoint(model_save_dir))

    output_graph_def = convert_variables_to_constants(
        sess,
        input_graph_def,
        output_node_names=['dense_decoded']
    )

    # last_compile_model_path = COMPILE_MODEL_PATH.replace('.pb', '_{}.pb'.format(int(acc * 10000)))
    # with tf.gfile.GFile(last_compile_model_path, mode='wb') as gf:
    #     gf.write(output_graph_def.SerializeToString())


def main():
    Trainer().train_process(RunMode.Trains)
    print('Training completed.')


if __name__ == '__main__':
    main()
