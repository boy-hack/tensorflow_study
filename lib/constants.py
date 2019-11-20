#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 2019/11/20 2:15 PM
# @Author  : w8ay
# @File    : constants.py
from enum import Enum, unique


@unique
class RunMode(Enum):
    Test = 'test'
    Trains = 'trains'
    Predict = 'predict'


@unique
class Optimizer(Enum):
    AdaBound = 'AdaBound'
    Adam = 'Adam'
    Momentum = 'Momentum'
    SGD = 'SGD'
    AdaGrad = 'AdaGrad'
    RMSProp = 'RMSProp'


OPTIMIZER_MAP = {
    'AdaBound': Optimizer.AdaBound,
    'Adam': Optimizer.Adam,
    'Momentum': Optimizer.Momentum,
    'SGD': Optimizer.SGD,
    'AdaGrad': Optimizer.AdaGrad,
    'RMSProp': Optimizer.RMSProp
}
