import numpy


class Layer:
    def __set_weights(self, weights):
        self.__weights = numpy.transpose(weights)

    def __get_weights(self):
        return self.__weights

    def __set_inputs(self, inputs):
        self.__inputs = inputs

    def __get_inputs(self):
        return self.__inputs

    def __init__(self, neurons_count, inputs_count):
        self.__init_weights(neurons_count, inputs_count)

    def __init_weights(self, neurons_count, inputs_count):
        weights = numpy.random.rand(neurons_count, inputs_count)
        self.__set_weights(weights)

    def input(self, inputs):
        if len(inputs) != len(self.weights):
            raise Exception(
                'Input size not correspond to layer neurons inputs count. ' + str(len(inputs)) + ' != ' + str(
                    len(self.weights)))
        self.inputs = inputs

        return numpy.array([Layer.activation(weighted_sum) for weighted_sum in numpy.dot(self.inputs, self.weights)])

    def correct_weights(self, deltas):
        derivatives_of_activation = numpy.array(
            [Layer.activation_derivative(weighted_sum) for weighted_sum in numpy.dot(self.inputs, self.weights)]
        )
        weights_transposed = numpy.transpose(self.weights)
        new_weights = numpy.array([neuron_weights +
                                   self.learning_speed *
                                   deltas[index] *
                                   derivatives_of_activation[index] *
                                   self.inputs
                                   for index, neuron_weights in enumerate(weights_transposed)])
        new_delta = numpy.dot(deltas, weights_transposed)
        self.weights = new_weights

        return new_delta

    @staticmethod
    def activation(x):
        return 1 / (1 + numpy.exp(-x))

    @staticmethod
    def activation_derivative(x):
        return Layer.activation(x) * (1 - Layer.activation(x))

    weights = property(__get_weights, __set_weights)
    inputs = property(__get_inputs, __set_inputs)
    learning_speed = 5
