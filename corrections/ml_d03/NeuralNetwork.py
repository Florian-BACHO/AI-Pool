from Neuron import *

class NeuralNetwork:
    def __init__(self, layerDescriptor):
        self.layers = []
        for i, it in enumerate(layerDescriptor):
            layer = self.createLayer(layerDescriptor[i], 0 if i == 0 else layerDescriptor[i - 1])
            self.layers.append(layer)

    def createLayer(self, nbNeuron, nbLast):
        return [Neuron(nbLast) for i in range(nbNeuron)]

    def activate(self, entries):
        for i in range(len(entries)):
            self.layers[0][i].out = entries[i] # Initialize entry layer with entries values
        for i in range(1, len(self.layers)): # For each layer
            for j in range(0, len(self.layers[i])): # For each neuron of this layer
                self.layers[i][j].activateWithLayer(self.layers[i - 1]) # Compute the neuron

    def getOutputs(self):
        out = []
        for i in self.layers[len(self.layers) - 1]:
            out.append(i.out)
        return out
