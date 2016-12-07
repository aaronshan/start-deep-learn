# _*_ coding:utf-8 _*_


class Perceptron(object):
    def __init__(self, input_num, activator):
        """
        初始化感知机
        :param input_num: 输入的参数个数
        :param activator: 激活函数 double -> double
        :return:
        """
        # 权重向量的所有值初始化为0
        self.weights = [0.0 for _ in range(input_num)]
        self.activator = activator
        # 偏置初始化为0
        self.bias = 0.0

    def predict(self, input_vector):
        """
        :param input_vector: 输入向量
        :return: 感知机的计算结果
        """
        # 将input_vector[x1, x2...]和weights[w1, w2....]打包在一起, 变成[(x1, w1), (x2, w2)...]
        # 用map来计算[x1*w1, x2*w2]
        multiply_result = map(lambda (x, w): x * w, zip(input_vector, self.weights))
        # 用reduce对打包后的结果求和
        reduce_result = reduce(lambda a, b: a + b, multiply_result, 0.0)
        return self.activator(reduce_result + self.bias)

    def iteration(self, input_vectors, labels, rate):
        """
        对所有数据进行一次迭代
        :param input_vectors: 训练数据
        :param labels: 正确的结果
        :param rate: 学习率
        :return:
        """
        # 将训练数据和正确结果打包在一起
        samples = zip(input_vectors, labels)
        # 对每个样本数据，按照感知器规则更新权重
        for input_vector, label in samples:
            # 先计算感知器在当前权重下的输出
            output = self.predict(input_vector)
            # 更新权重
            self.update_weight(input_vector, output, label, rate)

    def update_weight(self, input_vector, output, label, rate):
        """
        按照感知器规则更新权重
        :param input_vector: 输入向量
        :param output: 计算出的输出
        :param label: 正确的输出结果
        :param rate: 学习率
        :return:
        """
        delta = label - output
        # 将input_vector[x1, x2...]和weights[w1, w2....]打包在一起, 变成[(x1, w1), (x2, w2)...]
        self.weights = map(lambda (x, w): w + rate * delta * x, zip(input_vector, self.weights))
        # 更新bias
        self.bias += rate * delta

    def train(self, input_vectors, labels, iteration, rate):
        """
        输入训练数据，进行训练
        :param input_vectors: 训练数据
        :param labels: 正确的结果
        :param iteration: 迭代次数
        :param rate: 学习率
        :return:
        """
        for i in range(iteration):
            self.iteration(input_vectors, labels, rate)

    def __str__(self):
        """
        打印学习到的权重和偏置项
        :return:
        """
        return "weights\t:%s, bias\t:%f\n" % (self.weights, self.bias)