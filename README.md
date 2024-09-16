# Flappy Bird ML with Neural Networks and Genetic Algorithm

This project is a machine learning-based implementation of the Flappy Bird game, where the bird is controlled by a neural network. The game simulates an evolving population of birds using a genetic algorithm, with each generation learning to improve through mutations and crossover.

## Features
- **Neural Network Control**: Each bird in the game is controlled by a neural network that learns how to navigate the pipes.
- **Genetic Algorithm**: Birds evolve using a genetic algorithm with elitism, crossover, and mutation.
- **Fitness-Based Learning**: Birds are rewarded based on their performance, such as proximity to pipes and surviving longer.
- **Visual Feedback**: The game is displayed using Pygame, providing visual feedback on the birds’ performance and evolution.

## Project Structure

- **NeuralNetwork**: Defines the neural network architecture and handles feedforward operations using a tanh activation function.
- **Bird**: Represents a bird controlled by a neural network. Each bird has a fitness score that tracks its survival and performance.
- **Pipe**: Represents the obstacles (pipes) that the birds need to navigate.
- **Evolution Functions**: `evolve_population`, `crossover`, and `mutate` handle the genetic algorithm, including breeding and mutation of birds.
- **Game Loop**: Runs the game simulation, updating birds and pipes, rendering the game, and handling generation evolution.

## How to Run

1. Install the necessary dependencies:
    ```bash
    pip install pygame numpy
    ```

2. Run the game:
    ```bash
    python flappy_bird_ml.py
    ```

3. Use the game window to observe the evolution of the birds across generations.

## Configuration

You can modify the neural network and genetic algorithm parameters in the `bird_config` dictionary within the script. Parameters include:
- `num_inputs`: Number of input neurons (default: 4).
- `num_hidden`: Number of hidden neurons (default: 20).
- `num_outputs`: Number of output neurons (default: 1).
- `weight_init_mean`, `weight_init_stdev`: Mean and standard deviation for weight initialization.
- `bias_init_mean`, `bias_init_stdev`: Mean and standard deviation for bias initialization.
- `weight_mutate_power`, `bias_mutate_power`: Controls the magnitude of mutation for weights and biases during evolution.

## Gameplay

- Birds start at position `(50, 300)` and navigate a series of pipes.
- Birds decide whether to "jump" based on the output of their neural network, which processes information like the bird's current position, velocity, and the position of the nearest pipe.
- A generation ends when all birds have collided with pipes or the ground.
- Each bird's performance is measured using a fitness score, which helps the genetic algorithm evolve better birds for future generations.

## Fitness Function

- Birds are rewarded for:
    - Surviving longer (general fitness increase).
    - Approaching the nearest pipe.
    - Passing through pipes (large fitness boost).
    - An additional exponential multiplier is applied for higher scores.

## Customization

Feel free to modify the game by changing the following aspects:
- **Pipes**: Adjust the gap height or frequency to increase difficulty.
- **Neural Network Architecture**: Experiment with different numbers of hidden neurons or activation functions.
- **Genetic Algorithm**: Adjust the mutation rate, number of elites, or selection mechanism for more diverse or refined evolutionary behavior.

## Credits

Inspired by the original Flappy Bird game. This project showcases the application of neural networks and genetic algorithms in a game-based learning environment.

