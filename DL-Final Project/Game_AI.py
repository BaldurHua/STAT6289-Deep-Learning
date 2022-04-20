#%%[markdown]
# A modified version of snake_game.py in order to implement the AI

#%%
import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
# from package import text_object, message_display, button

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
# font = pygame.font.SysFont('arial', 25)  Longer run time

class Direction(Enum): # Set up position
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200,0,0)
GREEN1 = (0, 100, 0)
GREEN2 = (0, 155, 0)
BLACK = (0,0,0)
# GREY = (150,150,150)
# bGREEN = (0,200,0)

BLOCK_SIZE = 20 # Snake body size
SPEED = 40  # For bot higher the speed, shorter the runtime

class SnakeAI:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        # self.playSurface = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('Snake!')
        image = pygame.image.load('Snake.png')
        pygame.display.set_icon(image)
        self.reset()
        
    def reset(self):
        # init game state
        self.direction = Direction.RIGHT
        
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
    def _place_food(self):
        # Set up the food coordinate
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake: # Food not inside the snake
            self._place_food()
    
    # def pause_game(self):
    #     paused = True
    #     while paused:
    #         for event in pygame.event.get():  
    #             if event.type == pygame.QUIT:
    #                 pygame.quit()
    #                 quit()
    #             if event.type == pygame.KEYDOWN:
    #                 if event.key == pygame.K_c:
    #                     paused = False
    #                 elif event.key == pygame.K_q:
    #                     pygame.quit()
    #                     quit()
        
    def play_step(self, action):
        self.frame_iteration += 1
        # Player input (arrow key or WASD)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
                
            # if event.type == pygame.KEYDOWN:
            #     if event.key == pygame.K_LEFT or event.key == ord('a'):
            #         self.change_dir = Direction.LEFT
            #         if not self.direction == Direction.RIGHT:
            #             self.direction = self.change_dir
                        
            #     elif event.key == pygame.K_RIGHT or event.key == ord('d'):
            #         self.change_dir = Direction.RIGHT
            #         if not self.direction == Direction.LEFT:
            #             self.direction = self.change_dir
                        
            #     elif event.key == pygame.K_UP or event.key == ord('w'):
            #         self.change_dir = Direction.UP
            #         if not self.direction == Direction.DOWN:
            #             self.direction = self.change_dir
                    
            #     elif event.key == pygame.K_DOWN or event.key == ord('s'):
            #         self.change_dir = Direction.DOWN
            #         if not self.direction == Direction.UP:
            #             self.direction = self.change_dir
                    
            #     elif event.key == pygame.K_SPACE:
            #         self.pause_game()

        
        # movement
        self._move(action) 
        self.snake.insert(0, self.head)
        
        # check if game over
        reward = 0 # initial reward
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake): # later argument is to penalize the lack of imporvement
            game_over = True
            reward = -5
            return reward, game_over, self.score
            
        # place new food or just move
        if self.head == self.food:
            self.score += 1
            reward = 5 # reward for successfully eat the food
            self._place_food()
        else:
            self.snake.pop()
        
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # return game over and score
        return reward, game_over, self.score
    
    def is_collision(self, pt = None):
        if pt == None:
            pt = self.head
        # hits set-up boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        # hits itself except the head
        if pt in self.snake[1:]:
            return True
        
        return False
        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for pt in self.snake:
            pygame.draw.rect(self.display, GREEN1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, GREEN2, pygame.Rect(pt.x+4, pt.y+4, 12, 12))
            
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action): # AI direction [current direction, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx] 
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else: 
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx] 

        self.direction = new_dir
        
        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
        
    # def gameOver(self, score):
    #     gameOverFont = pygame.font.Font('arial.ttf', 72)
    #     gameOverSurf, gameOverRect = text_object('Game Over', gameOverFont, GREY)
    #     gameOverRect.midtop = (320, 125)
    #     self.playSurface.blit(gameOverSurf, gameOverRect)
    #     scoreFont = pygame.font.Font('arial.ttf', 48)
    #     scoreSurf, scoreRect = text_object('SCORE:'+str(score), scoreFont, GREY)
    #     scoreRect = scoreSurf.get_rect()
    #     scoreRect.midtop = (320, 225)
    #     self.playSurface.blit(scoreSurf, scoreRect)
    #     button(self.playSurface, 'Again', self.w//4, 
    #            self.h//8*7, self.w//2, 
    #            self.h//8, GREEN2, bGREEN, self.init_game)
    #     pygame.display.update()    
    
            

if __name__ == '__main__':
    game = SnakeAI()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
            break
        
    print('Final Score', score)
         
    pygame.quit()