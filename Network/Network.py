class Network:
    def __set_layers(self, layers):
        self.__layers = layers

    def __get_layers(self):
        return self.__layers

    def __init__(self, layers):
        self.__set_layers(layers)

    def input(self, inputs):
        if len(inputs) != len(self.layers[0].weights):
            raise Exception('Inputs not correspond to input layer. Expected ' + str(len(self.layers[0].weights)) + ', got ' + str(len(inputs)))

        output = inputs
        for layer in self.layers:
            output = layer.input(output)
        return output

    def learning(self, training_set):
        error = 5

        while error > 0.25:
            for training_data in training_set:
                last_index = len(training_data) - 1
                correct_output = training_data[last_index]

                output = self.input(training_data[0:last_index])
                delta = correct_output - output

                for i in delta:
                    error = i * i

                for layer in reversed(self.layers):
                    delta = layer.correct_weights(delta)
            print(error)

    layers = property(__get_layers, __set_layers)
