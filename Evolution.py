# evolution.py

import random
import numpy as np
import copy
from bird import Bird
from neural_network import NeuralNetwork

def select_parents(birds):
    total_fitness = sum(bird.fitness for bird in birds)
    if total_fitness == 0:
        selection_probs = [1 / len(birds)] * len(birds)
    else:
        selection_probs = [bird.fitness / total_fitness for bird in birds]
    parent1, parent2 = random.choices(birds, weights=selection_probs, k=2)
    return parent1, parent2

def evolve_population(birds, config, mutation_rate=0.05, num_elites=2):
    birds.sort(key=lambda b: b.fitness, reverse=True)
    num_elites = min(num_elites, len(birds))
    new_birds = [Bird(config, nn=copy.deepcopy(birds[i].nn)) for i in range(num_elites)]
    total_fitness = sum(bird.fitness for bird in birds)
    if total_fitness == 0:
        selection_probs = [1 / len(birds)] * len(birds)
    else:
        selection_probs = [bird.fitness / total_fitness for bird in birds]
    while len(new_birds) < len(birds):
        parent1, parent2 = random.choices(birds, weights=selection_probs, k=2)
        child_nn = crossover(parent1.nn, parent2.nn, config)
        mutate(child_nn, config, mutation_rate)
        new_birds.append(Bird(config, nn=child_nn))
    return new_birds

def crossover(nn1, nn2, config):
    child_nn = NeuralNetwork(config)
    # Uniform crossover for weights_input_hidden
    mask = np.random.rand(*nn1.weights_input_hidden.shape) > 0.5
    child_nn.weights_input_hidden = np.where(mask, nn1.weights_input_hidden, nn2.weights_input_hidden)
    # Uniform crossover for weights_hidden_output
    mask = np.random.rand(*nn1.weights_hidden_output.shape) > 0.5
    child_nn.weights_hidden_output = np.where(mask, nn1.weights_hidden_output, nn2.weights_hidden_output)
    # Average biases
    child_nn.bias_hidden = (nn1.bias_hidden + nn2.bias_hidden) / 2
    child_nn.bias_output = (nn1.bias_output + nn2.bias_output) / 2
    return child_nn

def mutate(nn, config, mutation_rate):
    # Mutate weights_input_hidden
    mutation_mask = np.random.rand(*nn.weights_input_hidden.shape) < mutation_rate
    nn.weights_input_hidden += mutation_mask * np.random.normal(0, config['weight_mutate_power'], nn.weights_input_hidden.shape)
    # Mutate weights_hidden_output
    mutation_mask = np.random.rand(*nn.weights_hidden_output.shape) < mutation_rate
    nn.weights_hidden_output += mutation_mask * np.random.normal(0, config['weight_mutate_power'], nn.weights_hidden_output.shape)
    # Mutate bias_hidden
    mutation_mask = np.random.rand(*nn.bias_hidden.shape) < mutation_rate
    nn.bias_hidden += mutation_mask * np.random.normal(0, config['bias_mutate_power'], nn.bias_hidden.shape)
    # Mutate bias_output
    mutation_mask = np.random.rand(*nn.bias_output.shape) < mutation_rate
    nn.bias_output += mutation_mask * np.random.normal(0, config['bias_mutate_power'], nn.bias_output.shape)
    # Clip weights and biases to prevent exploding values
    nn.weights_input_hidden = np.clip(nn.weights_input_hidden, -1, 1)
    nn.weights_hidden_output = np.clip(nn.weights_hidden_output, -1, 1)
    nn.bias_hidden = np.clip(nn.bias_hidden, -1, 1)
    nn.bias_output = np.clip(nn.bias_output, -1, 1)
