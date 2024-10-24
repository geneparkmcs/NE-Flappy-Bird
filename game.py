# game.py

import pygame
import random
from bird import Bird
from pipe import Pipe
from evolution import evolve_population
from config import bird_config

def game_loop(config):
    pygame.init()
    screen = pygame.display.set_mode((400, 600))
    pygame.display.set_caption("Flappy Bird ML")
    WHITE = (255, 255, 255)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    
    population_size = 200
    birds = [Bird(config) for _ in range(population_size)]
    all_birds = birds.copy()  # Keep track of all birds in the generation
    
    pipes = [Pipe(400), Pipe(600)]
    generation = 1
    score = 0
    running = True
    clock = pygame.time.Clock()
    
    font = pygame.font.SysFont(None, 36)
    
    while running:
        screen.fill(WHITE)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        alive_birds = False  # Flag to check if any bird is alive
        for bird in birds:
            if bird.alive:
                alive_birds = True
                bird.update()
                bird.decide(pipes)
                if bird.check_collision(pipes):
                    bird.alive = False
        
        if not alive_birds:
            print(f"Generation {generation} ended. Score: {score}. Evolving next generation...")
            birds = evolve_population(all_birds, config)
            all_birds = birds.copy()
            generation += 1
            pipes = [Pipe(400), Pipe(600)]
            score = 0
            continue  # Skip the rest of the loop to start the next generation
        
        for pipe in pipes:
            pipe.update()
        
        if pipes[-1].x < 250:
            pipes.append(Pipe(400))
        
        for pipe in pipes[:]:
            if pipe.x + pipe.width < 0:
                pipes.remove(pipe)
            if not pipe.passed and pipe.x + pipe.width < 50:
                pipe.passed = True
                score += 1
        
        # Draw pipes
        for pipe in pipes:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pipe.x, 0, pipe.width, pipe.gap_y))
            pygame.draw.rect(screen, GREEN, pygame.Rect(pipe.x, pipe.gap_y + pipe.gap_height, pipe.width, 600))
        
        # Draw the best bird
        alive_birds_list = [bird for bird in birds if bird.alive]
        if alive_birds_list:
            # Sort alive birds by fitness to find the best one
            alive_birds_list.sort(key=lambda b: b.fitness, reverse=True)
            best_bird = alive_birds_list[0]
            pygame.draw.rect(screen, RED, pygame.Rect(best_bird.x, int(best_bird.y), best_bird.width, best_bird.height))
        
        # Display score and generation
        score_text = font.render(f"Score: {score}  Generation: {generation}", True, (0, 0, 0))
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(120)  # Increase FPS to speed up training
    pygame.quit()

if __name__ == "__main__":
    game_loop(bird_config)
