from Matrix import Matrix
import math

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def dsigmoid(y):
    # This assumes y is x that has already been sigmoided.
    return y * (1 - y)

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Weights between input nodes and hidden nodes
        self.weights_ih = Matrix(self.hidden_nodes, self.input_nodes)
        # Weights between hidden nodes and output nodes
        self.weights_ho = Matrix(self.output_nodes, self.hidden_nodes)
        self.weights_ih.randomise()
        self.weights_ho.randomise()

        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        self.bias_h.randomise()
        self.bias_o.randomise()

        self.learning_rate = 0.1

    def feedforward(self, inp_array):
        # Generate hidden layer output
        inputs = Matrix.fromArray(inp_array) # Make a one-column matrix (2D) from a 1D array
        hidden = Matrix.multiply(self.weights_ih, inputs) # Matrix product of input and weights
        hidden.add(self.bias_h) # Add bias to prevent '0' results
        hidden.map(sigmoid) # Activation Function - Applies sigmoid (essentially normalisation) to the matrix.

        # Generate output layer output
        output = Matrix.multiply(self.weights_ho, hidden)
        output.add(self.bias_o)
        output.map(sigmoid)

        return output.toArray()

    def train(self, inp_array, target_array):
        # Generate hidden layer output
        inputs = Matrix.fromArray(inp_array) # Make a one-column matrix (2D) from a 1D array
        hidden = Matrix.multiply(self.weights_ih, inputs) # Matrix product of input and weights
        hidden.add(self.bias_h) # Add bias to prevent '0' results
        hidden.map(sigmoid) # Activation Function - Applies sigmoid (essentially normalisation) to the matrix.

        # Generate output layer output
        outputs = Matrix.multiply(self.weights_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)
        
        # Convert arrays to matrix object
        targets = Matrix.fromArray(target_array)

        # Calculate total error
        output_errors = Matrix.subtract(targets, outputs)

        # Calculate gradient
        gradients = Matrix.mapMatrix(outputs, dsigmoid)
        gradients.mult(output_errors)
        gradients.mult(self.learning_rate)

        # Calculate deltas
        hidden_transposed = Matrix.transpose(hidden)
        weights_ho_deltas = Matrix.multiply(gradients, hidden_transposed)

        # Apply deltas to weights
        self.weights_ho.add(weights_ho_deltas)

        # Adjust the bias by it's deltas (the gradients)
        self.bias_o.add(gradients)
        
        # Calculate hidden layer errors (error per perceptron)
        weights_ho_transposed = Matrix.transpose(self.weights_ho)
        hidden_errors = Matrix.multiply(weights_ho_transposed, output_errors)

        # Calculate hidden layer gradients
        hidden_gradients = Matrix.mapMatrix(hidden, dsigmoid)
        hidden_gradients.mult(hidden_errors)
        hidden_gradients.mult(self.learning_rate)

        # Calculate input layer -> hidden layer deltas
        inputs_transposed = Matrix.transpose(inputs)
        weights_ih_deltas = Matrix.multiply(hidden_gradients, inputs_transposed)

        # Apply deltas to weights
        self.weights_ih.add(weights_ih_deltas)

        # Adjust the hidden bias by it's deltas (the hidden gradients)
        self.bias_h.add(hidden_gradients)
