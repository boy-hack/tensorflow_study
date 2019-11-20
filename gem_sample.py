#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 11:21 AM
# @Author  : w8ay
# @File    : gem_sample.py.py

import os
import random
import time

# 验证码样本生成工具
from captcha.image import ImageCaptcha

import conf.sample_conf as conf


def gen_special_img(text, file_path, width, height):
    # 生成img文件
    generator = ImageCaptcha(width=width, height=height)  # 指定大小
    img = generator.generate_image(text)  # 生成图片
    img.save(file_path)  # 保存图片


def gen_ima_by_batch(root_dir, image_suffix, characters, count, char_count, width, height):
    # 判断文件夹是否存在
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for index in range(count):
        text = ""
        for j in range(char_count):
            text += random.choice(characters)

        timec = str(time.time()).replace(".", "")
        p = os.path.join(root_dir, "{}_{}.{}".format(text, timec, image_suffix))
        gen_special_img(text, p, width, height)

        print("Generate captcha image => {}".format(index + 1))


def main():
    gen_ima_by_batch(conf.root_dir, conf.image_suffix, conf.characters, conf.count, conf.char_count, conf.width,
                     conf.height)


if __name__ == '__main__':
    main()
