# _*_ coding:utf-8 _*_


class LinearUnit(object):
    def __init__(self, input_num, activator):
        """
        初始化线性单元, 设置输入参数个数
        :param input_num: 输入参数的个数
        :param activator: 激活函数
        :return:
        """
        self.weights = [1.0 for _ in range(input_num)]
        self.activator = activator
        self.bias = 1.0

    def predict(self, input_vector):
        """
        :param input_vector: 输入向量
        :return: 线性单元的计算结果
        """
        # 将input_vector[x1, x2...]和weights[w1, w2....]打包在一起, 变成[(x1, w1), (x2, w2)...]
        # 用map来计算[x1*w1, x2*w2]
        multiply_result = map(lambda (x, w): x * w, zip(input_vector, self.weights))
        # 用reduce对打包后的结果求和
        reduce_result = reduce(lambda a, b: a + b, multiply_result, 0.0)
        return self.activator(reduce_result + self.bias)

    def batch_gradient_descend(self, input_vectors, labels, rate):
        """
        批梯度下降算法
        :param input_vectors: 输入向量
        :param labels: 正确的结果
        :param rate: 学习率
        :return:
        """
        # 将训练数据和正确结果打包在一起
        samples = zip(input_vectors, labels)
        input_num = self.weights.__len__()
        sums = [0.0 for _ in range(input_num)]
        bias_tmp = 0.0
        for input_vector, label in samples:
            # 先计算线性单元在当前权重下的输出
            output = self.predict(input_vector)
            losses = [(label - output) for _ in range(input_num)]
            rates = [rate for _ in range(input_num)]
            sums = map(lambda w, r, loss, x_i: w + r * loss * x_i, sums, rates, losses, input_vector)
            bias_tmp += rate*(label - output)

        self.weights = map(lambda x, y: x + y, self.weights, sums)
        self.bias += bias_tmp

    def train(self, input_vectors, labels, iteration, rate):
        """
        对训练数据集进行训练
        :param input_vectors: 输入向量
        :param labels: 正确的结果
        :param iteration: 迭代次数
        :param rate: 学习率
        :return:
        """
        for i in range(iteration):
            self.batch_gradient_descend(input_vectors, labels, rate)

    def __str__(self):
        """
        打印学习到的权重和偏置项
        :return:
        """
        return "weights\t:%s, bias\t:%f\n" % (self.weights, self.bias)
