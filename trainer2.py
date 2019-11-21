import os

import cv2
import numpy as np

from conf.trainer_conf import image_width, image_height, train_image_dir, test_image_dir, train_batch_size, \
    test_batch_size, model_save_dir, GEN_CHAR_SET

CHARS = GEN_CHAR_SET
CHARS_DICT = {char: i for i, char in enumerate(CHARS)}
# width：图片宽度；height：图片高度；n_len：图片字数；n_class：一共分为多少个类（要多加一个类，因为图片有的地方没输出）
width, height, n_len, n_class = image_width, image_height, 4, len(CHARS) + 1
epoch = 1


# CTC 是一种损失函数，它用来衡量输入的序列数据经过神经网络之后，和真实的输出相差有多少。主要用来文字识别和语音识别
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def encode_label(s):
    label = np.zeros([len(s)])
    for i, c in enumerate(s):
        label[i] = CHARS_DICT[c]
    return label


# 解析生成的标签文本
def parse_line(line):
    # parts = line.split(':')
    # filename = parts[0]
    # label = encode_label(parts[1].strip())
    filename = line
    label = os.path.basename(line).split("_")[0]
    return filename, label


from keras.models import *
from keras.layers import *

# rnn_size 神经元个数，一般常用 32、64、128、256、512但是并不是越多越好，可以多多测试这个是训练调参主要参数（要在比较复杂的项目效果才能体现）。
rnn_size = 512
# base_conv 卷积层的filters个数，也是主要调参参数（要在比较复杂的项目效果才能体现）。
base_conv = 32
input_tensor = Input((width, height, 3))
x = input_tensor
for i in range(3):
    # 卷积层，（3,3）是卷积核大小，一般是奇数常用有：3、5、7；数字越大对图片视野越大，但是专家测试过了3是最好的
    x = Conv2D(base_conv * (2 ** (i)), (3, 3))(x)
    # 归一化
    x = BatchNormalization()(x)
    # 激活函数relu
    x = Activation('relu')(x)
    # 池化层
    x = MaxPooling2D(pool_size=(2, 2))(x)
conv_shape = x.get_shape()
# 把多维数据平坦化，简单就是说把多维数据转化为一维数据
x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2] * conv_shape[3])))(x)
# 隐藏层神经元有32个
x = Dense(32, activation='relu')(x)
# 2个GRU添加，然后在2个GRU合并，使用了双向循环神经网络，一正一反
gru_1 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru1_b')(x)
gru1_merged = add([gru_1, gru_1b])

gru_2 = GRU(rnn_size, return_sequences=True, init='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, init='he_normal', name='gru2_b')(gru1_merged)
x = concatenate([gru_2, gru_2b])
# 防止过拟合
x = Dropout(0.25)(x)
# 输出层 激活函数是softmax
x = Dense(n_class, init='he_normal', activation='softmax')(x)
base_model = Model(input=input_tensor, output=x)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=[1], dtype='int64')
label_length = Input(name='label_length', shape=[1], dtype='int64')
loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

model = Model(input=[input_tensor, labels, input_length, label_length], output=[loss_out])
# 定义训练方式，ctc是损失函数，optimizer是优化器
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')


# 批量给出训练数据
class TextImageGenerator:
    def __init__(self, img_dir, batch_size, img_size, num_channels=3, label_len=4):
        self._img_dir = img_dir
        self._batch_size = batch_size
        self._num_channels = num_channels
        self._label_len = label_len
        self._img_w, self._img_h = img_size

        self._num_examples = 0
        self._next_index = 0
        self._num_epoches = 0
        self.filenames = []
        self.labels = None

        self.init()

    def init(self):
        self.labels = []
        paths = os.listdir(self._img_dir)
        for i in paths:
            path = os.path.join(os.path.realpath(self._img_dir), i)
            self.filenames.append(path)
            label = os.path.basename(path).split("_")[0]
            self.labels.append(encode_label(label))
            self._num_examples += 1
        self.labels = np.float32(self.labels)

    def next_batch(self):
        # Shuffle the data
        if self._next_index == 0:
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._filenames = [self.filenames[i] for i in perm]
            self._labels = self.labels[perm]

        batch_size = self._batch_size
        start = self._next_index
        end = self._next_index + batch_size
        if end >= self._num_examples:
            self._next_index = 0
            self._num_epoches += 1
            end = self._num_examples
            batch_size = self._num_examples - start
        else:
            self._next_index = end
        # images = np.zeros([batch_size, self._img_h, self._img_w, self._num_channels])
        X = np.zeros((batch_size, self._img_w, self._img_h, self._num_channels), dtype=np.uint8)
        # labels = np.zeros([batch_size, self._label_len])
        y = np.zeros((batch_size, self._label_len), dtype=np.uint8)
        for j, i in enumerate(range(start, end)):
            fname = self._filenames[i]
            img = cv2.imread(fname)
            try:
                X[j] = img.transpose(1, 0, 2)
            except BaseException as e:
                print(fname)
                print(str(e))
        labels = self._labels[start:end, ...]
        return [X, labels, np.ones(batch_size) * int(conv_shape[1] - 2), np.ones(batch_size) * n_len], np.ones(
            batch_size)

    def get_data(self):
        while True:
            yield self.next_batch()


from keras.utils import plot_model


# 查看模型摘要，会生成model.png图片，这个是全部网络模型
# plot_model(model, to_file='model.png', show_shapes=True)


# 主要对验证集测试准确度
def evaluate(model, batch_num=40):
    batch_acc = 0
    # 定义给出的数据类，img_dir：图片路径；label_file：图片对应的标签路径；batch_size：一次生成多少个数据；
    # img_size：图片的宽度和高度；num_channels：图片通道数，3是彩色图片，1是灰色图片
    generator = TextImageGenerator(img_dir=test_image_dir,
                                   batch_size=test_batch_size,
                                   img_size=(image_width, image_height),
                                   num_channels=3,
                                   label_len=4).get_data()
    print('保存模型')
    global epoch
    model_dir = os.path.join(os.path.realpath(model_save_dir), "verification_mode_{}.h5".format(epoch))
    # 保存模型
    base_model.save(model_dir)
    epoch = epoch + 1
    for i in range(batch_num):
        # 生成数据
        [X_test, y_test, _, _], _ = next(generator)
        # 预测和真实标签对比，给定验证集识别率acc
        y_pred = base_model.predict(X_test)
        shape = y_pred[:, 2:, :].shape
        out = K.get_value(K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:, :4]
        if out.shape[1] == 4:
            batch_acc += ((y_test == out).sum(axis=1) == 4).mean()
    return batch_acc / batch_num


from keras.callbacks import *


# 主要对验证集测试准确度
class Evaluate(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(base_model) * 100
        self.accs.append(acc)
        print("")
        print('acc: %f%%' % acc)


evaluator = Evaluate()


def train():
    # 模型训练，samples_per_epoch：几个批量；nb_epoch：迭代几次，每次迭代图片数量 = samples_per_epoch * batch_size
    # callbacks：回调函数，每次迭代一次回调一次evaluator函数来看验证集准确度，patience：容忍度，也就是训练如果没有在学习到东西，
    #            在20次就会自动停止
    # validation_data，验证集
    model.fit_generator(TextImageGenerator(img_dir=train_image_dir,
                                           batch_size=train_batch_size,
                                           img_size=(image_width, image_height),
                                           num_channels=3,
                                           label_len=4).get_data(), samples_per_epoch=100, nb_epoch=300,
                        callbacks=[EarlyStopping(patience=20), evaluator],
                        validation_data=TextImageGenerator(img_dir=test_image_dir,
                                                           batch_size=test_batch_size,
                                                           img_size=(image_width, image_height),
                                                           num_channels=3,
                                                           label_len=4).get_data(), nb_val_samples=40)


if __name__ == '__main__':
    # 训练
    train()
