import numpy as np

class NeuralNetwork:
    def __init__(self, config):
        self.num_inputs = config['num_inputs']
        self.num_hidden = config['num_hidden']
        self.num_outputs = config['num_outputs']
        
        self.weights_input_hidden = np.random.normal(config['weight_init_mean'], config['weight_init_stdev'], (self.num_inputs, self.num_hidden))
        self.weights_hidden_output = np.random.normal(config['weight_init_mean'], config['weight_init_stdev'], (self.num_hidden, self.num_outputs))
        
        self.bias_hidden = np.random.normal(config['bias_init_mean'], config['bias_init_stdev'], self.num_hidden)
        self.bias_output = np.random.normal(config['bias_init_mean'], config['bias_init_stdev'], self.num_outputs)
    
    def activation(self, x):
        return np.tanh(x)
    
    def feedforward(self, inputs):
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.activation(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        return self.activation(final_input)