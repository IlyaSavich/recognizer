import numpy
from Network.Layer import Layer
from Network.Network import Network

# a = numpy.array([1])
# b = numpy.array([[1, 2], [3, 4], [5, 6]])
# print(b)
layer1 = Layer(2, 2)
layer2 = Layer(1, 2)
network = Network([layer1, layer2])

training_set = numpy.array([
    [0, 0, numpy.array([0])],
    [1, 0, numpy.array([1])],
    [0, 1, numpy.array([1])],
    [1, 1, numpy.array([0])],
], dtype=object)
network.learning(training_set)

# for layer in network.layers:
#     print(layer.weights)

print('###################')
print(network.input([0, 0]))
print(network.input([0, 1]))
print(network.input([1, 0]))
print(network.input([1, 1]))
