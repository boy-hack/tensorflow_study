#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 11:34 AM
# @Author  : w8ay
# @File    : trainer_conf.py

# 样本基本配置
import string
from lib.constants import RunMode as runmode

origin_image_dir = "sample/origin/"
train_image_dir = "sample/train/"
test_image_dir = "sample/test/"
model_save_dir = "model/"
image_width = 100
image_height = 60
image_suffix = "png"
# Todo 自动处理origin目录内的的图片，将它们缩放到设置大小，图片格式转换为image_suffix格式

# 训练配置
characters = string.ascii_lowercase + string.digits  # 字符集
cycle_stop = 10000  # 循环多少步停止
acc_stop = 0.99  # 准确率达到多少停止
cycle_save = 500  # 每多少步保存一次模型
enable_gpu = False  # 是否启用GPU训练
test_percentage = 0.05  # 分配百分比的数据给测试集

# 神经网络相关参数
train_batch_size = 128  # 训练批次大小
test_batch_size = 100  # 测试批次大小
IMAGE_CHANNEL = 1  # 颜色通道，有些网络需要有些网络不需要，1为黑白，3为RGB彩色
TRAINS_LEARNING_RATE = 0.1  # 学习率
DECAY_STEPS = 10000
DECAY_RATE = 0.98
NEU_OPTIMIZER = 'AdaBound'
MOMENTUM = 0.9
CTC_TOP_PATHS = 1

# LSTM 网络配置
NUM_HIDDEN = 64
LSTM_LAYER_NUM = 2

# CTC 网络配置
CTC_MERGE_REPEATED = True
CTC_BEAM_WIDTH = 1
PREPROCESS_COLLAPSE_REPEATED = False
CTC_LOSS_TIME_MAJOR = True

# 下面由程序更改，无需自己配置
RunMode = runmode.Trains
RESIZE = [image_width, image_height]
GEN_CHAR_SET = [str(i) for i in characters]
CHAR_SET_LEN = len(GEN_CHAR_SET)
NUM_CLASSES = CHAR_SET_LEN + 2
