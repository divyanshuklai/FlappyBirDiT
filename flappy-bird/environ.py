# Original game code from https://codewithcurious.com/projects/flappy-bird-game-using-python/
# Code heavily modified
# Importing the libraries
import pygame
import os
import gymnasium as gym
import numpy as np

from pathlib import Path

curpath = Path(__file__).parent

class FlappyBirdEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps":120}
    PIPE_WIDTH = 69
    PIPE_HEIGHT = 425

    def __init__(self, render_mode=None):

        self.render_mode = render_mode
        pygame.init()

        self.width = 350
        self.height = 622
        self.gravity = 0.17

        self.action_space = gym.spaces.Discrete(2) # 0:nothing 1:flap
        self.observation_space = gym.spaces.Box(
            low =-np.inf, high=np.inf, shape=(4,), dtype=np.float32 
        )

        self.game_over = True #reset to begin
        self.truncated = False
        self.score = 0
        self.high_score = 0

        self.bird_rect = pygame.Rect(67, 311, 34, 24)
        self.bird_movement = 0
        self.bird_index = 0


        self.pipes = []

        self.floor_x = 0

        self.bird_flap = pygame.USEREVENT
        pygame.time.set_timer(self.bird_flap, 200)

        self.create_pipe = pygame.USEREVENT + 1
        pygame.time.set_timer(self.create_pipe, 1200)

        self.quit_signal = False

        # Game window
        if self.render_mode in self.metadata["render_modes"]:
            os.environ['SDL_VIDEO_WINDOW_POS'] = '0,0'
            os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
            os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"     

            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Flappy Bird")

            self.clock = pygame.time.Clock()

            # setting background and base image
            self.back_img = pygame.image.load(curpath / "img_46.png")
            self.floor_img = pygame.image.load(curpath / "img_50.png")

            # different stages of bird
            bird_up = pygame.image.load(curpath / "img_47.png")
            bird_down = pygame.image.load(curpath / "img_48.png")
            bird_mid = pygame.image.load(curpath / "img_49.png")
            self.birds = [bird_up, bird_mid, bird_down]
            self.bird_img = self.birds[self.bird_index]

            # Loading pipe image
            self.pipe_img = pygame.image.load(curpath /"greenpipe.png")

            # for the pipes to appear
            pygame.time.set_timer(self.create_pipe, 1200)

            # Displaying game over image
            self.over_img = pygame.image.load(curpath / "img_45.png").convert_alpha ()
            self.over_rect = self.over_img.get_rect(center=(self.width // 2, self.height // 2))

            # setting variables and font for score
            self.score_font = pygame.font.Font("freesansbold.ttf", 27)

            # end game if running for too long

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game_over = False
        self.truncated = False
        self.quit_signal = False

        self.score = 0

        self.bird_movement = 0
        self.bird_rect.center = (67, 311)

        self.pipes = []

        return self._get_obs(), self._get_info()
    
    def step(self, action):
        if action == 1 and self.bird_movement >= -3.5:
            self.bird_movement = -7

        pipe_crossed = self._update_state()

        reward = 0
        if not self.game_over:
            reward += 0.01
        else:
            reward += -1.0
        if pipe_crossed:
            reward += 1.0
        # if abs(self.bird_movement) > 7:
        #     reward -= 0.1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, self.game_over, self.truncated, info
    
    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return None
        
        self._render_frame()

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            frame = pygame.surfarray.array3d(self.screen)
            return np.transpose(frame, (1, 0, 2))
        
        return None

    def close(self):
        if self.render_mode in self.metadata["render_modes"]:
            pygame.display.quit()
            pygame.quit()

    def _get_obs(self):
        bird_y = self.bird_rect.centery
        bird_velocity = self.bird_movement
        next_window_x, next_window_y = 600, 600
        for pipe in self.pipes:
            if pipe["top"].right > self.bird_rect.left:
                next_window_x = pipe['top'].centerx - self.bird_rect.centerx
                next_window_y = (pipe['top'].bottom + pipe['bottom'].top) // 2
                break
        if next_window_x < self.height:
            return np.array(
                [bird_y / self.height, bird_velocity, next_window_x / self.width, next_window_y /self.height],
                dtype=np.float32
            )
        else:
            return np.array(
                [bird_y / self.height, bird_velocity, 600 / self.width, 600 /self.height],
                dtype=np.float32
            )
    
    def _get_info(self):
        return {"score" : self.score}
    
    def _update_state(self):
        # events
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                self.quit_signal = True
                return False # Return immediately to avoid processing other events

            if event.type == self.bird_flap and not self.game_over:
                self.bird_index = (self.bird_index + 1) % len(self.birds)

            if ( event.type == self.create_pipe and not self.game_over
            and not any(pipe["top"].right > 250 for pipe in self.pipes) ):
                gap_center_y = np.random.randint(150, 350)
                pipe_gap = np.random.randint(200, 300)
                start_x = 440 + np.random.randint(0, 60)
                top_pipe = pygame.Rect(
                    start_x, 
                    gap_center_y - (pipe_gap// 2) - self.PIPE_HEIGHT,
                    self.PIPE_WIDTH,
                    self.PIPE_HEIGHT
                )
                bottom_pipe = pygame.Rect(
                    start_x,
                    gap_center_y + (pipe_gap // 2),
                    self.PIPE_WIDTH,
                    self.PIPE_HEIGHT
                )
                self.pipes.append({
                    "top":top_pipe,
                    "bottom":bottom_pipe,
                    "scored":False
                })

        if not self.game_over:
            #bird
            self.bird_movement += self.gravity
            self.bird_rect.centery += self.bird_movement

            #pipes
            for pipe in self.pipes:
                pipe["top"].centerx -= 3
                pipe["bottom"].centerx -= 3
                if pipe["top"].right < 0:
                    self.pipes.remove(pipe)

            # collisions
                if (self.bird_rect.colliderect(pipe["top"]) 
                    or self.bird_rect.colliderect(pipe["bottom"])):
                    self.game_over = True
            if self.bird_rect.bottom < 0 or self.bird_rect.bottom >= 510:
                self.game_over = True

            # truncated
            if self.score > 500:
                self.truncated = True

            # score
            pipe_crossed = False
            if self.pipes:
                for pipe in self.pipes:
                    if (not pipe['scored'] and 
                    pipe['top'].centerx < self.bird_rect.centerx - 2):
                        self.score += 1
                        pipe['scored'] = True
                        pipe_crossed = True
            
            if self.score > self.high_score:
                self.high_score = self.score

            # floor
            self.floor_x -= 1
            if self.floor_x < -448:
                self.floor_x = 0
            
            return pipe_crossed

        return False

    def _render_frame(self):

        # blit bg and floor
        self.screen.blit(self.back_img, (0, 0))
        self.screen.blit(self.floor_img, (self.floor_x, 520))
        self.screen.blit(self.floor_img, (self.floor_x + 448, 520))

        # blit pipes
        for pipe in self.pipes:
            flipped_pipe = pygame.transform.flip(self.pipe_img, False, True)
            self.screen.blit(flipped_pipe, pipe["top"])
            self.screen.blit(self.pipe_img, pipe["bottom"])

        # blit bird
        self.bird_img = self.birds[self.bird_index]
        rotated_bird = pygame.transform.rotozoom(self.bird_img, self.bird_movement * -6, 1)
        self.screen.blit(rotated_bird, self.bird_rect)

        # blit score
        if self.game_over:
            score_text = self.score_font.render(f" Score: {self.score}", True, (255, 255, 255))
            score_rect = score_text.get_rect(center=(self.width // 2, 66))
            self.screen.blit(score_text, score_rect)

            high_score_text = self.score_font.render(f"High Score: {self.high_score}", True, (255, 255, 255))
            high_score_rect = high_score_text.get_rect(center=(self.width // 2, 506))
            self.screen.blit(high_score_text, high_score_rect)
        else:
            score_text = self.score_font.render(str(self.score), True, (255, 255, 255))
            score_rect = score_text.get_rect(center=(self.width // 2, 66))
            self.screen.blit(score_text, score_rect)

    def get_action_meanings(self):
        return {0: "", 1: "space"}


