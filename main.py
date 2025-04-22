import pygame
import pygame.camera
import sys
import os
import math
import time
import numpy as np
import cv2
from backend.GameLogic import GameLogic

pygame.init()
FPS = 60
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
RED = (255, 0, 0)
ATTACK_DISPLAY_DURATION = 4.0


# game states
START = "START"
INITIALIZE = "INITIALIZE"
RUNNING = "RUNNING"
END = "END"

ASSET_FOLDER_PATH = "assets"
BACKGROUND_FOLDER_PATH = "assets/background"
FONTS_FOLDER_PATH = "assets/fonts"
OBJECTS_FOLDER_PATH = "assets/objects"


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("StreetFighterLIVE")
        self.clock = pygame.time.Clock()

        # initialize backend game
        self.backend = GameLogic()
        self.state = START
        self.attack_timer = 0
        self.attack = None
        self.info_font = None
        
        self.load_start_assets()
        

    def load_start_assets(self):
        try:
            self.start_bg = pygame.image.load(
                f"{BACKGROUND_FOLDER_PATH}/start.png"
            )
            self.start_bg = pygame.transform.scale(
                self.start_bg, (SCREEN_WIDTH, SCREEN_HEIGHT)
            )

            self.inital_bg = pygame.image.load(
                f"{BACKGROUND_FOLDER_PATH}/initalize.png"
            )
            self.inital_bg = pygame.transform.scale(
                self.inital_bg, (SCREEN_WIDTH, SCREEN_HEIGHT)
            )

            self.title_img = pygame.image.load(f"{ASSET_FOLDER_PATH}/title/title.png")

            title_width = self.title_img.get_width()
            title_height = self.title_img.get_height()

            scale_factor = 0.8
            self.title_img = pygame.transform.scale(
                self.title_img,
                (int(title_width * scale_factor), int(title_height * scale_factor)),
            )
            self.title_rect = self.title_img.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            )

            try:
                self.info_font = pygame.font.Font(f'{FONTS_FOLDER_PATH}/PressStart2P-Regular.ttf', 24)
            except:
                self.info_font = pygame.font.Font(None, 18)
        except pygame.error as e:
            print(f"ERROR loading images: {e}")
            return

    def initialize_game(self, singleplayer=True):
        if singleplayer:
            self.load_single_player_assets()
        else:
            self.load_multiplayer_assets() 
        
        # self.camera = cv2.VideoCapture(0)
        # if not self.camera.isOpened():
        #     raise RuntimeError(f"Could not open video capture device {0}")
        
        self.backend.start()
        self.state = RUNNING
    
    def load_single_player_assets(self):
        self.running_bg = pygame.image.load(
                f"{ASSET_FOLDER_PATH}/background/gym.jpg"
            )
         
        self.running_bg = pygame.transform.scale(
            self.running_bg, (SCREEN_WIDTH//2, SCREEN_HEIGHT)
        )

        self.punching_bag = pygame.image.load(
            f"{OBJECTS_FOLDER_PATH}/punching_bag.png"
        )

        punching_bag_height = int(SCREEN_HEIGHT * 0.6)  # 60% of screen height
        punching_bag_width = int(self.punching_bag.get_width() * 
                                (punching_bag_height / self.punching_bag.get_height()))
        
        self.punching_bag = pygame.transform.scale(
            self.punching_bag, (punching_bag_width, punching_bag_height)
        )
        
        # Position it on the left side of the screen
        self.punching_bag_rect = self.punching_bag.get_rect(
            center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
        )
    

    def load_multiplayer_assets(self):
        return NotImplementedError

    def render_start(self):
        self.screen.blit(self.start_bg, (0, 0))
        self.screen.blit(self.title_img, self.title_rect)

    def render_initialize(self):
        self.screen.blit(self.inital_bg, (0, 0))
    
    def render_running(self, frame):
        self.screen.blit(self.running_bg, (0, 0))
        self.screen.blit(self.punching_bag, self.punching_bag_rect)

        if frame is None: 
            print("Error: No Camera Initialized.")
        else: # CENTER THE CAMERA
            h, w = frame.shape[:2]
            target_aspect = (SCREEN_WIDTH/2) / SCREEN_HEIGHT
            
            if w/h > target_aspect:
                new_w = int(h * target_aspect)
                start_x = (w - new_w) // 2
                frame = frame[:, start_x:start_x+new_w]
            else:
                new_h = int(w / target_aspect)
                start_y = (h - new_h) // 2
                frame = frame[start_y:start_y+new_h, :]
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            surface = pygame.image.frombuffer(
                np.ascontiguousarray(frame_rgb).tobytes(), 
                frame_rgb.shape[1::-1], 
                'RGB'
            )
            

        
            surface = pygame.transform.scale(
                surface, (SCREEN_WIDTH // 2, SCREEN_HEIGHT)
            )

            self.screen.blit(surface, (SCREEN_WIDTH // 2, 0))

        # Display attack text if we have one
        current_time = time.time()
        if self.attack and current_time < self.attack_timer:
            attack_text = self.info_font.render(self.attack, True, RED)
            attack_rect = attack_text.get_rect(center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4))
            self.screen.blit(attack_text, attack_rect)
    
    def render_end(self):
        return

    def render(self, frame):
        if self.state == START:
            self.render_start()
        elif self.state == INITIALIZE:
            self.render_initialize()
        elif self.state == RUNNING:
            self.render_running(frame)
        elif self.state == END:
            self.render_end()

    def handle_events(self, input):
        if input != None:
            if self.state == START and input:
                self.state = INITIALIZE
                print("STARTING SINGLEPLAYER") 
                self.initialize_game(True)               
            elif self.state == START and not input:
                self.state = INITIALIZE
                print("STARTING MULTIPLAYER")
                self.initialize_game(False)
            elif self.state == RUNNING and input:
                self.attack = input['AttackType']
                self.attack_timer = time.time() + ATTACK_DISPLAY_DURATION

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()

                if self.state == START and event.key == pygame.K_SPACE:
                    self.state = INITIALIZE
                elif self.state == INITIALIZE and event.key == pygame.K_SPACE:
                    self.load_game_assets()
                    self.state = RUNNING
                elif self.state == END and event.key == pygame.K_r:
                    self.currentstate_state = START

    def run(self):
        while True:
            input = self.backend.read()
            print(input)
            self.handle_events(input)
            # Update game state
            # self.update()
            frame = self.backend.get_frame()
            self.render(frame)

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    game = Game()
    game.run()
