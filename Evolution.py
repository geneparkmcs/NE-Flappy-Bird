import random
import numpy as np
from Bird import Bird
from NeuralNetwork import NeuralNetwork

def evolve_population(birds, config, mutation_rate=0.05, num_elites=5):
    birds.sort(key=lambda b: b.fitness, reverse=True)

    new_birds = []
    while len(new_birds) < len(birds) - num_elites:
        parent1, parent2 = random.choices(birds[:num_elites], k=2)
        child_nn = crossover(parent1.nn, parent2.nn, config)
        mutate(child_nn, config, mutation_rate)
        new_birds.append(Bird(config, nn=child_nn))
    
    new_birds += [Bird(config, nn=elite.nn) for elite in new_birds]

    return new_birds

def crossover(nn1, nn2, config):
    child_nn = NeuralNetwork(config)
    
    for i in range(nn1.weights_input_hidden.shape[0]):
        for j in range(nn1.weights_input_hidden.shape[1]):
            child_nn.weights_input_hidden[i][j] = random.choice([nn1.weights_input_hidden[i][j], nn2.weights_input_hidden[i][j]])
    
    for i in range(nn1.weights_hidden_output.shape[0]):
        for j in range(nn1.weights_hidden_output.shape[1]):
            child_nn.weights_hidden_output[i][j] = random.choice([nn1.weights_hidden_output[i][j], nn2.weights_hidden_output[i][j]])

    child_nn.bias_hidden = random.choice([nn1.bias_hidden, nn2.bias_hidden])
    child_nn.bias_output = random.choice([nn1.bias_output, nn2.bias_output])
    
    return child_nn

def mutate(nn, config, mutation_rate):
    for i in range(nn.weights_input_hidden.shape[0]):
        for j in range(nn.weights_input_hidden.shape[1]):
            if random.random() < mutation_rate:
                nn.weights_input_hidden[i][j] += np.random.normal(0, config['weight_mutate_power'])
    
    for i in range(nn.weights_hidden_output.shape[0]):
        for j in range(nn.weights_hidden_output.shape[1]):
            if random.random() < mutation_rate:
                nn.weights_hidden_output[i][j] += np.random.normal(0, config['weight_mutate_power'])
    
    if random.random() < mutation_rate:
        nn.bias_hidden += np.random.normal(0, config['bias_mutate_power'])
    
    if random.random() < mutation_rate:
        nn.bias_output += np.random.normal(0, config['bias_mutate_power'])
