# _*_ coding:utf-8 _*_

from LinearUnit import *
from perceptron import Activator


def get_train_data():
    """
    真实函数为y=2x_1^2 + x_2 + 3
    :return: 返回数据
    """
    # 输入向量
    train_inputs = [[1, 2], [2, 1], [2, 3], [3, 5], [1, 3], [4, 2], [7, 3], [4, 5], [11, 3], [8, 7]]
    # 输入向量对应的值
    train_labels = [7, 8, 10, 14, 8, 13, 20, 16, 28, 26]

    return train_inputs, train_labels


def train_and_linear_unit():
    """
    使用训练集来训练线性单元
    :return: 训练好的线性单元
    """
    # 创建一个线性单元，输入参数为2个，激活函数选择relu
    l = LinearUnit(2, Activator.relu)
    # 开始训练，迭代1000次，学习率为0.001
    input_vectors, labels = get_train_data()
    l.train(input_vectors, labels, 10000, 0.001)
    # 返回训练好
    return l

if __name__ == '__main__':
    # 训练感知机
    l = train_and_linear_unit()
    print l
    # 测试
    test_inputs = [[1, 4], [2, 2], [2, 5], [5, 3], [1, 5], [4, 1]]
    test_labels = [9, 9, 12, 16, 10, 12]
    for test_input, test_label in zip(test_inputs, test_labels):
        x, y = test_input
        print "%f and %f => %f, label=%f" % (x, y, l.predict([x, y]), test_label)
