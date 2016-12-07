# _*_ coding:utf-8 _*_


import Activator
from Perceptron import *


def get_train_data():
    """
    :return: 返回and真值表的数据
    """
    # 输入向量
    input_vectors = [[1, 1], [1, 0], [0, 1], [0, 0]]
    # 输入向量对应的真值[[1, 1] => 1, [1, 0] => 0, [0, 1] => 0, [0, 0] => 0]
    labels = [1, 0, 0, 0]

    return input_vectors, labels


def train_and_perceptron():
    """
    使用and真值表来训练感知机
    :return: 训练好的感知机
    """
    # 创建一个感知机，输入参数为2个（and是2元函数），激活函数选择relu
    p = Perceptron(2, Activator.f)
    # 开始训练，迭代100次，学习率为0.01
    input_vectors, labels = get_train_data()
    p.train(input_vectors, labels, 1000, 0.01)
    # 返回训练好的感知机
    return p

if __name__ == '__main__':
    # 训练感知机
    and_perceptron = train_and_perceptron()
    print and_perceptron
    # 测试
    print "1 and 1 => %f" % and_perceptron.predict([1, 1])
    print "1 and 0 => %f" % and_perceptron.predict([1, 0])
    print "0 and 1 => %f" % and_perceptron.predict([0, 1])
    print "0 and 0 => %f" % and_perceptron.predict([0, 0])
