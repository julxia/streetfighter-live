import pygame
import pygame.camera
import sys
import os
import math
import time
from backend.GameLogic import GameLogic

pygame.init()
FPS = 60
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900

# game states
START = "START"
INITIALIZE = "INITIALIZE"
RUNNING = "RUNNING"
END = "END"

ASSET_FOLDER_PATH = "assets"


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("StreetFighterLIVE")
        self.clock = pygame.time.Clock()

        # initialize backend game
        self.backend = GameLogic()
        self.state = START

        self.load_start_assets()

        return

    def load_start_assets(self):
        try:
            self.start_bg = pygame.image.load(
                f"{ASSET_FOLDER_PATH}/background/start.png"
            )
            self.start_bg = pygame.transform.scale(
                self.start_bg, (SCREEN_WIDTH, SCREEN_HEIGHT)
            )

            self.inital_bg = pygame.image.load(
                f"{ASSET_FOLDER_PATH}/background/initalize.png"
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
        except pygame.error as e:
            print(f"ERROR loading images: {e}")
            return

    def load_single_player_assets(self):
        return

    def load_multiplayer_assets(self):
        return

    def render_start(self):
        self.screen.blit(self.start_bg, (0, 0))
        self.screen.blit(self.title_img, self.title_rect)

    def render_initialize(self):
        self.screen.blit(self.inital_bg, (0, 0))

    def render_running(self):
        return

    def render_end(self):
        return

    def render(self):
        if self.state == START:
            self.render_start()
        elif self.state == INITIALIZE:
            self.render_initialize()
        elif self.state == RUNNING:
            self.render_running()
        elif self.state == END:
            self.render_end()

    def handle_events(self, input):
        if input != None:
            if self.state == START and input:
                self.load_single_player_assets()
                print("STARTING SINGLEPLAYER")
            elif self.state == START and not input:
                self.load_multiplayer_assets()
                print("STARTING MULTIPLAYER")

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
            self.handle_events(input)
            # Update game state
            # self.update()

            self.render()

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    game = Game()
    game.run()
