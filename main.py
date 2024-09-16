import numpy as np
import pygame
import random

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

def evolve_population(birds, config, mutation_rate=0.05, num_elites=2):
    birds.sort(key=lambda b: b.fitness, reverse=True)

    new_birds = []
    while len(new_birds) < len(birds) - num_elites:
        parent1, parent2 = random.choices(birds[:2], k=2)
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

class Pipe:
    def __init__(self, x, gap_y, gap_height=150):
        self.x = x
        self.gap_y = gap_y
        self.gap_height = gap_height
        self.width = 50
        self.passed = False

    def update(self):
        self.x -= 5

def game_loop(config):
    pygame.init()
    screen = pygame.display.set_mode((400, 600))
    pygame.display.set_caption("Flappy Bird ML")

    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    
    population_size = 200  
    birds = [Bird(config) for _ in range(population_size)]
    
    pipes = [Pipe(400, 200), Pipe(600, 180)]
    generation = 1
    score = 0
    running = True
    clock = pygame.time.Clock()
    
    while running:
        screen.fill(WHITE)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        if birds:
            for bird in birds:
                bird.update()
                bird.decide(pipes)
                if bird.check_collision(pipes):
                    birds.remove(bird)
                nearest_pipe = bird.get_nearest_pipe(pipes)
                if nearest_pipe:
                    bird.fitness += 1.0 / (nearest_pipe.x - bird.x + 1)
                    center_of_gap = nearest_pipe.gap_y + nearest_pipe.gap_height / 2
                    y_distance = abs(center_of_gap - bird.y)
                    bird.fitness += max(0, 50 - y_distance)
        
        for pipe in pipes:
            pipe.update()

        if pipes[-1].x < 250:
            pipes.append(Pipe(400, random.randint(150, 400)))
        
        for pipe in pipes:
            if pipe.x + pipe.width < 0:
                pipes.remove(pipe)
            if not pipe.passed and pipe.x < 50:
                score += 1
                pipe.passed = True
                
                for bird in birds:
                    bird.fitness += 30
                    if score > 0 and score % 10 == 0:
                        multiplier = 2 ** (score // 10)
                        bird.fitness *= multiplier

        for pipe in pipes:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pipe.x, 0, pipe.width, pipe.gap_y))
            pygame.draw.rect(screen, GREEN, pygame.Rect(pipe.x, pipe.gap_y + pipe.gap_height, pipe.width, 600))

        for bird in birds:
            pygame.draw.rect(screen, RED, pygame.Rect(bird.x, int(bird.y), bird.width, bird.height))

        if not birds:
            print(f"Generation {generation} ended. Score: {score}. Evolving next generation...")
            birds = evolve_population([Bird(config) for _ in range(population_size)], config)
            generation += 1
            pipes = [Pipe(400, 200), Pipe(600, 180)]
            score = 0

        font = pygame.font.SysFont(None, 36)
        score_text = font.render(f"Score: {score}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))

        pygame.display.flip()
        clock.tick(100)

    pygame.quit()

if __name__ == "__main__":
    bird_config = {
        'num_inputs': 4,
        'num_hidden': 20,
        'num_outputs': 1,
        'weight_init_mean': 0.0,
        'weight_init_stdev': 0.5,
        'bias_init_mean': 0.0,
        'bias_init_stdev': 0.5,
        'weight_mutate_power': .01,
        'bias_mutate_power': .01
    }
    game_loop(bird_config)
