# _*_ coding:utf-8 _*_

import math


def sigmoid(x):
    return 1.0 / (1.0 + math.pow(math.e, -x))


def relu(x):
    return x if x > 0 else 0


def tanh(x):
    e_x = math.pow(math.e, x)
    e_n_x = math.pow(math.e, -x)
    return (e_x - e_n_x) / (e_x + e_n_x)


def f(x):
    return 1 if x > 0 else 0