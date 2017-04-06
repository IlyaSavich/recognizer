import numpy


class Layer:
    def __set_weights(self, weights):
        self.__weights = numpy.transpose(weights)

    def __get_weights(self):
        return self.__weights

    def __init__(self, neurons_count, inputs_count):
        self.__init_weights(neurons_count, inputs_count)

    def __init_weights(self, neurons_count, inputs_count):
        weights = numpy.random.rand(neurons_count, inputs_count)
        self.__set_weights(weights)

    def input(self, inputs):
        if len(inputs) != len(self.weights):
            raise Exception('Input size not correspond to layer neurons inputs count. ' + str(len(inputs)) + ' != ' + str(len(self.weights)))
        return [Layer.activation(weighted_sum) for weighted_sum in numpy.dot(inputs, self.weights)]

    @staticmethod
    def activation(x):
        return 1 / (1 + numpy.exp(-x))

    weights = property(__get_weights, __set_weights)
