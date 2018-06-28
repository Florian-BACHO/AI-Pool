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

    # Calculate the total loss of the network
    def calcLoss(self, dataset):
        loss = 0.
        for ex in dataset.exemples:
            self.activate(ex.inputs)
            for i in range(len(self.layers[-1])):
                loss += ex.outputs[i] * log(self.layers[-1][i].out) +\
                        (1. - ex.outputs[i]) * log(1. - self.layers[-1][i].out)
        return -loss / len(dataset.exemples)

    # NOT ADAPTED TO MULTI-LAYER NEURAL NETWORKS !!!
    def calculateGradients(self, desiredOutputs):
        for i in range(len(self.layers[len(self.layers) - 1])):
            self.layers[len(self.layers) - 1][i].calculateOutputGradient(desiredOutputs[i])

    def applyGradients(self, learningRate):
        for i in range(1, len(self.layers)):
            for n in self.layers[i]:
                n.applyGradientWithLayer(learningRate, self.layers[i - 1])
