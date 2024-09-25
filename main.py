import numpy as np
import pygame
import random
from Bird import Bird
from Evolution import evolve_population
from Pipe import Pipe

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
        clock.tick(800)

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
