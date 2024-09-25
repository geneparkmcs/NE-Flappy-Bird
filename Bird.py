from NeuralNetwork import NeuralNetwork
import numpy as np
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
        self.high_score_multiplier = 0
        self.nn = nn if nn else NeuralNetwork(config)
    
    def update(self):
        self.velocity += self.gravity
        self.y += self.velocity
        self.fitness += 0.1

    def decide(self, pipes):
        nearest_pipe = self.get_nearest_pipe(pipes)
        if nearest_pipe:
            inputs = np.array([self.y, self.velocity, nearest_pipe.x - self.x, nearest_pipe.gap_y])
            output = self.nn.feedforward(inputs)
            if output > 0.5:
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
        for pipe in pipes:
            if (pipe.x < self.x + self.width < pipe.x + pipe.width) or (pipe.x < self.x < pipe.x + pipe.width):
                if not (pipe.gap_y < self.y < pipe.gap_y + pipe.gap_height):
                    return True
        return self.y < 0 or self.y > 600