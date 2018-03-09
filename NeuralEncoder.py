import numpy
from json import JSONEncoder
from NeuralNetwork import ActivationFunction

class NeuralEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        if isinstance(obj, ActivationFunction):
            return None
        return JSONEncoder.default(self, obj)
