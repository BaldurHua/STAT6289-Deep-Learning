#%%[markdown]
# Snake Game
#
# [Reference](https://github.com/kiteco/python-youtube-code/tree/master/snake) 
# To run this game. Open Anaconda prompt and change the current directory all the way to the folder of this py file. Then, python snake_game.py

#%%
import pygame
import random
from enum import Enum
from collections import namedtuple
from package import text_object, message_display, button

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
GREY = (150,150,150)
bGREEN = (0,200,0)

BLOCK_SIZE = 20 # Snake body size
SPEED = 10

class SnakeGame:
    
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # init display
        self.display = pygame.display.set_mode((self.w, self.h))
        self.playSurface = pygame.display.set_mode((self.w, self.h))
        self.clock = pygame.time.Clock()
        self.init_game()
        
        # init game state
    def init_game(self):
        self.direction = Direction.RIGHT
        self.change_dir = self.direction
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-(2*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        
    def _place_food(self):
        # Set up the food coordinate
        x = random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE 
        y = random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake: # Food not inside the snake
            self._place_food()
        
    def pause_game(self):
        paused = True
        while paused:
            for event in pygame.event.get():  
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_c:
                        paused = False
                    elif event.key == pygame.K_q:
                        pygame.quit()
                        quit()
            p_text = font.render("Press c to continue or q to quit" , True, WHITE)
            self.display.blit(p_text, [0, 50])
            pygame.display.flip()
        
    def play_step(self):
        # Player input (arrow key or WASD)
        for event in pygame.event.get():  
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT or event.key == ord('a'):
                    self.change_dir = Direction.LEFT
                    if not self.direction == Direction.RIGHT:
                        self.direction = self.change_dir
                        
                elif event.key == pygame.K_RIGHT or event.key == ord('d'):
                    self.change_dir = Direction.RIGHT
                    if not self.direction == Direction.LEFT:
                        self.direction = self.change_dir
                        
                elif event.key == pygame.K_UP or event.key == ord('w'):
                    self.change_dir = Direction.UP
                    if not self.direction == Direction.DOWN:
                        self.direction = self.change_dir
                    
                elif event.key == pygame.K_DOWN or event.key == ord('s'):
                    self.change_dir = Direction.DOWN
                    if not self.direction == Direction.UP:
                        self.direction = self.change_dir
                    
                elif event.key == pygame.K_SPACE:
                    self.pause_game()

        # movement
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)
        # check if game over
        game_over = False
        if self._is_collision():
            game_over = True
            return game_over, self.score
        
        # place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()
        # update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # return game over and score
        return game_over, self.score
    
    def _is_collision(self):
        # hits set-up boundary
        if self.head.x > self.w - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.h - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits itself except the head
        if self.head in self.snake[1:]:
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
        
    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)
        
    def gameOver(self, score):
        gameOverFont = pygame.font.Font('arial.ttf', 72)
        gameOverSurf, gameOverRect = text_object('Game Over', gameOverFont, GREY)
        gameOverRect.midtop = (320, 125)
        self.playSurface.blit(gameOverSurf, gameOverRect)
        scoreFont = pygame.font.Font('arial.ttf', 48)
        scoreSurf, scoreRect = text_object('SCORE:'+str(score), scoreFont, GREY)
        scoreRect = scoreSurf.get_rect()
        scoreRect.midtop = (320, 225)
        self.playSurface.blit(scoreSurf, scoreRect)
        button(self.playSurface, 'Again', self.w//4, 
               self.h//8*7, self.w//2, 
               self.h//8, GREEN2, bGREEN, self.init_game)
        pygame.display.update()
        
    def play():
        pygame.display.set_caption('Snake!')
        image = pygame.image.load('Snake.png')
        pygame.display.set_icon(image)

        pygame.init()
        game = SnakeGame()
        while True:
            score, done = game.play_step()
            if done:
                game.gameOver(score)
            

if __name__ == '__main__':
    pygame.display.set_caption('Snake!')
    # Load image and set icon
    image = pygame.image.load('Snake.png')
    pygame.display.set_icon(image)

    pygame.init()
    game = SnakeGame()
    
    # game loop
    while True:
        game_over, score = game.play_step()
        
        if game_over == True:
             game.gameOver(score)
        
    # print('Final Score', score)
        
    # pygame.quit()
# %%
