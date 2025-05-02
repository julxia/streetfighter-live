import pygame
import pygame.camera
import sys
import os
import math
import time
import numpy as np
import cv2
from backend.GameLogic import GameLogic
from multiplayer.network_client import NetworkClient

from dotenv import load_dotenv

load_dotenv()

pygame.init()
FPS = 60
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
RED = (255, 0, 0)
WHITE = (255, 255, 255)

ATTACK_DISPLAY_DURATION = 1.0


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

        # Multiplayer Changes
        # Initialize network client for multiplayer
        self.network = NetworkClient(host=os.getenv("IPV4_ADDR"))

        # initialize backend game
        self.backend = GameLogic(multiplayer_client=self.network)
        self.state = START
        self.attack_timer = 0
        self.attack = None
        # Multiplayer Changes
        self.opponent_attack = None
        self.opponent_attack_timer = 0
        self.info_font = None

        # Multiplayer Changes
        # Health bars
        self.player_health = 100
        self.opponent_health = 100

        self.init_start_time = 0
        self.init_duration = 2.0

        self.load_start_assets()

    def load_start_assets(self):
        try:
            self.start_bg = pygame.image.load(f"{BACKGROUND_FOLDER_PATH}/start.png")
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
            self.title_y_offset = 0
            self.animation_time = 0
            scale_factor = 0.8
            self.title_img = pygame.transform.scale(
                self.title_img,
                (int(title_width * scale_factor), int(title_height * scale_factor)),
            )
            self.title_rect = self.title_img.get_rect(
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
            )

            self.title_y_pos = SCREEN_HEIGHT // 2

            try:
                self.info_font = pygame.font.Font(
                    f"{FONTS_FOLDER_PATH}/PressStart2P-Regular.ttf", 30
                )
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

    def load_single_player_assets(self):
        self.running_bg = pygame.image.load(f"{ASSET_FOLDER_PATH}/background/gym.jpg")

        self.running_bg = pygame.transform.scale(
            self.running_bg, (SCREEN_WIDTH // 2, SCREEN_HEIGHT)
        )

        self.punching_bag = pygame.image.load(f"{OBJECTS_FOLDER_PATH}/punching_bag.png")

        punching_bag_height = int(SCREEN_HEIGHT * 0.6)  # 60% of screen height
        punching_bag_width = int(
            self.punching_bag.get_width()
            * (punching_bag_height / self.punching_bag.get_height())
        )

        self.punching_bag = pygame.transform.scale(
            self.punching_bag, (punching_bag_width, punching_bag_height)
        )

        # Position it on the left side of the screen
        self.punching_bag_rect = self.punching_bag.get_rect(
            center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
        )

    def load_multiplayer_assets(self):
        # Multiplayer Changes
        self.running_bg = pygame.image.load(f"{ASSET_FOLDER_PATH}/background/gym.jpg")

        self.running_bg = pygame.transform.scale(
            self.running_bg, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        # Health bar assets
        self.health_bar_width = 300
        self.health_bar_height = 30
        self.health_bar_border = 2

    def render_start(self):

        self.screen.blit(self.start_bg, (0, 0))
        self.screen.blit(self.title_img, self.title_rect)

        start_text = self.info_font.render(
            "Say 'Single Player' to start the game", True, WHITE
        )
        start_rect = start_text.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT * 3.5 // 4)
        )
        self.screen.blit(start_text, start_rect)

    def render_initialize(self):
        self.screen.blit(self.inital_bg, (0, 0))
        init_text = self.info_font.render("Starting Single Player mode...", True, WHITE)
        init_rect = init_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(init_text, init_rect)

    def render_running(self, frame):
        # Multiplayer Changes
        if self.backend.isSinglePlayer:
            # Single player rendering
            self.screen.blit(self.running_bg, (0, 0))
            self.screen.blit(self.punching_bag, self.punching_bag_rect)

            if frame is None:
                print("Error: No Camera Initialized.")
            else:  # CENTER THE CAMERA
                h, w = frame.shape[:2]
                target_aspect = (SCREEN_WIDTH / 2) / SCREEN_HEIGHT

                if w / h > target_aspect:
                    new_w = int(h * target_aspect)
                    start_x = (w - new_w) // 2
                    frame = frame[:, start_x : start_x + new_w]
                else:
                    new_h = int(w / target_aspect)
                    start_y = (h - new_h) // 2
                    frame = frame[start_y : start_y + new_h, :]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                surface = pygame.image.frombuffer(
                    np.ascontiguousarray(frame_rgb).tobytes(),
                    frame_rgb.shape[1::-1],
                    "RGB",
                )

                surface = pygame.transform.scale(
                    surface, (SCREEN_WIDTH // 2, SCREEN_HEIGHT)
                )

                self.screen.blit(surface, (SCREEN_WIDTH // 2, 0))

            # Display attack text if we have one
            current_time = time.time()
            if self.attack and current_time < self.attack_timer:
                attack_text = self.info_font.render(self.attack, True, WHITE)
                attack_rect = attack_text.get_rect(
                    center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT * 3 // 4)
                )
                self.screen.blit(attack_text, attack_rect)
        else:
            # Multiplayer Changes
            # Multiplayer rendering
            self.screen.blit(self.running_bg, (0, 0))

            # Process and display player frame
            if frame is not None:
                # Process player frame
                h, w = frame.shape[:2]
                target_aspect = (SCREEN_WIDTH / 2) / SCREEN_HEIGHT

                if w / h > target_aspect:
                    new_w = int(h * target_aspect)
                    start_x = (w - new_w) // 2
                    frame = frame[:, start_x : start_x + new_w]
                else:
                    new_h = int(w / target_aspect)
                    start_y = (h - new_h) // 2
                    frame = frame[start_y : start_y + new_h, :]

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                player_surface = pygame.image.frombuffer(
                    np.ascontiguousarray(frame_rgb).tobytes(),
                    frame_rgb.shape[1::-1],
                    "RGB",
                )

                player_surface = pygame.transform.scale(
                    player_surface, (SCREEN_WIDTH // 2, SCREEN_HEIGHT)
                )

                # Display player on left side
                self.screen.blit(player_surface, (0, 0))

            # Get and display opponent frame
            opponent_frame = self.backend.get_opponent_frame()
            if opponent_frame is not None:
                opponent_frame_rgb = cv2.cvtColor(opponent_frame, cv2.COLOR_BGR2RGB)
                opponent_surface = pygame.image.frombuffer(
                    np.ascontiguousarray(opponent_frame_rgb).tobytes(),
                    opponent_frame_rgb.shape[1::-1],
                    "RGB",
                )

                opponent_surface = pygame.transform.scale(
                    opponent_surface, (SCREEN_WIDTH // 2, SCREEN_HEIGHT)
                )

                # Display opponent on right side
                self.screen.blit(opponent_surface, (SCREEN_WIDTH // 2, 0))

            # Draw health bars
            self.player_health, self.opponent_health = (
                self.backend.multiplayer_client.get_health()
            )

            # Player health bar (left side)
            health_percent = max(0, self.player_health / 100)
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (50, 50, self.health_bar_width, self.health_bar_height),
            )
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (
                    50,
                    50,
                    int(self.health_bar_width * health_percent),
                    self.health_bar_height,
                ),
            )

            # Opponent health bar (right side)
            health_percent = max(0, self.opponent_health / 100)
            pygame.draw.rect(
                self.screen,
                (255, 0, 0),
                (
                    SCREEN_WIDTH - 50 - self.health_bar_width,
                    50,
                    self.health_bar_width,
                    self.health_bar_height,
                ),
            )
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                (
                    SCREEN_WIDTH - 50 - self.health_bar_width,
                    50,
                    int(self.health_bar_width * health_percent),
                    self.health_bar_height,
                ),
            )

            # Display attack text if we have one
            current_time = time.time()
            if self.attack and current_time < self.attack_timer:
                attack_text = self.info_font.render(self.attack, True, WHITE)
                attack_rect = attack_text.get_rect(
                    center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT * 3 // 4)
                )
                self.screen.blit(attack_text, attack_rect)

            # Display opponent attack if they performed one
            if self.opponent_attack and current_time < self.opponent_attack_timer:
                opponent_text = self.info_font.render(self.opponent_attack, True, WHITE)
                opponent_rect = opponent_text.get_rect(
                    center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4)
                )
                self.screen.blit(opponent_text, opponent_rect)

    def render_end(self):
        return NotImplementedError

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
        current_time = time.time()

        if input != None:
            # Game mode selection logic
            if self.state == START:
                # Multiplayer Changes
                if input is True:  # Single player
                    self.state = INITIALIZE
                    print("STARTING SINGLEPLAYER")
                    self.initialize_game(True)
                    self.init_start_time = time.time()
                elif input is False:  # Multiplayer
                    self.state = INITIALIZE
                    print("STARTING MULTIPLAYER")
                    self.initialize_game(False)
                    self.init_start_time = time.time()

            # Game running logic
            elif self.state == RUNNING and input and "AttackType" in input:
                if self.attack is None or current_time >= self.attack_timer:
                    self.attack = input["AttackType"]
                    self.attack_timer = current_time + ATTACK_DISPLAY_DURATION

                # Process health updates in multiplayer
                if not self.backend.isSinglePlayer:
                    if "player_health" in input:
                        self.player_health = input["player_health"]
                    if "opponent_health" in input:
                        self.opponent_health = input["opponent_health"]

                # Process opponent action
                if "opponent_action" in input and input["opponent_action"]:
                    opponent_action = input["opponent_action"]
                    if (
                        "AttackType" in opponent_action
                        and opponent_action["AttackType"]
                    ):
                        self.opponent_attack = opponent_action["AttackType"]
                        self.opponent_attack_timer = (
                            current_time + ATTACK_DISPLAY_DURATION
                        )

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
                    self.init_start_time = time.time()
                elif self.state == INITIALIZE and event.key == pygame.K_SPACE:
                    self.load_game_assets()
                elif self.state == END and event.key == pygame.K_r:
                    self.currentstate_state = START

    def update(self):
        self.animation_time += 0.05
        current_time = time.time()

        if self.state == START:
            self.title_y_offset = math.sin(self.animation_time) * 15
            self.title_rect.centery = self.title_y_pos + self.title_y_offset

        if (
            self.state == INITIALIZE
            and time.time() - self.init_start_time >= self.init_duration
        ):
            self.state = RUNNING

        if self.attack and current_time >= self.attack_timer:
            self.attack = None

    def run(self):
        while True:
            input = self.backend.read()
            # print(input)
            self.handle_events(input)
            # Update game state
            self.update()
            frame = self.backend.get_frame()
            self.render(frame)

            pygame.display.flip()
            self.clock.tick(FPS)


if __name__ == "__main__":
    game = Game()
    game.run()
