# bird.py

import numpy as np
from neural_network import NeuralNetwork

class Bird:
    def __init__(self, config, nn=None):
        self.x = 50
        self.y = 300
        self.velocity = 0
        self.gravity = 1
        self.lift = -12
        self.width = 20
        self.height = 20
        self.fitness = 0
        self.score = 0
        self.nn = nn if nn else NeuralNetwork(config)
        self.alive = True  # Add an alive flag
    
    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity
        self.fitness += 1  # Reward for staying alive
    
    def decide(self, pipes):
        nearest_pipe = self.get_nearest_pipe(pipes)
        if nearest_pipe:
            # Normalize inputs
            inputs = np.array([
                self.y / 600,  # Normalize y position
                self.velocity / 20,  # Normalize velocity
                (nearest_pipe.x - self.x) / 400,  # Normalize horizontal distance
                nearest_pipe.gap_y / 600  # Normalize gap y position
            ])
            output = self.nn.feedforward(inputs)
            if output > 0.0:
                self.jump()
    
    def jump(self):
        self.velocity = self.lift
    
    def get_nearest_pipe(self, pipes):
        nearest_pipe = None
        min_distance = float('inf')
        for pipe in pipes:
            distance = pipe.x - self.x
            if 0 < distance < min_distance:
                nearest_pipe = pipe
                min_distance = distance
        return nearest_pipe
    
    def check_collision(self, pipes):
        # Check if bird is out of screen bounds
        if self.y < 0 or self.y + self.height > 600:
            return True
        # Check collision with pipes
        for pipe in pipes:
            if self.x + self.width > pipe.x and self.x < pipe.x + pipe.width:
                if self.y < pipe.gap_y or self.y + self.height > pipe.gap_y + pipe.gap_height:
                    return True
        return False
