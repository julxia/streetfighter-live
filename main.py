import pygame
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
pygame.mixer.init()

FPS = 60
SCREEN_WIDTH = 1440
SCREEN_HEIGHT = 900
RED = (255, 0, 0)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
LILAC = (230,215,255)

ATTACK_DISPLAY_DURATION = 1.0


# game states
START = "START"
INITIALIZE = "INITIALIZE"
RUNNING = "RUNNING"
TERMINATE = "TERMINATE"

ASSET_FOLDER_PATH = "assets"
ATTACK_FOLDER_PATH = f"{ASSET_FOLDER_PATH}/attacks"
BACKGROUND_FOLDER_PATH = f"{ASSET_FOLDER_PATH}/background"
FONTS_FOLDER_PATH = f"{ASSET_FOLDER_PATH}/fonts"
OBJECTS_FOLDER_PATH = f"{ASSET_FOLDER_PATH}/objects"
MUSIC_FOLDER_PATH = f"{ASSET_FOLDER_PATH}/music"


class Game:
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("StreetFighterLIVE")
        self.clock = pygame.time.Clock()

        self.music_paths = {
            "general": f"{MUSIC_FOLDER_PATH}/general.mp3",
            "singleplayer": f"{MUSIC_FOLDER_PATH}/singleplayer.mp3",
            "multiplayer": f"{MUSIC_FOLDER_PATH}/multiplayer.mp3",
        }
        self.current_music = None 
        self.attack_sounds = {}
        self.attack_channels = {}
        self.attack_restore_time = None 

        # network client for multiplayer
        self.network = NetworkClient(host=os.getenv("IPV4_ADDR"))

        # initialize backend game
        self.backend = GameLogic(multiplayer_client=self.network)
        self.state = START
        self.attack_timer = 0
        self.attack = None

        # Multiplayer
        self.opponent_attack = None
        self.opponent_attack_timer = 0
        self.info_font = None
        self.player_health = 100
        self.opponent_health = 100

        self.init_start_time = 0
        self.init_duration = 5.0
        self.winner = None

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
            self.initial_tutorial = pygame.image.load(
                f"{OBJECTS_FOLDER_PATH}/tutorial.png"
            )

            tutorial_height = int(SCREEN_HEIGHT * 0.8)
            tutorial_width = int(
                self.initial_tutorial.get_width()
                * (tutorial_height / self.initial_tutorial.get_height())
            )

            self.initial_tutorial = pygame.transform.scale(
                self.initial_tutorial, (tutorial_width, tutorial_height)
            )

            self.initial_tutorial_rect = self.initial_tutorial.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
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
                center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT * 0.9)
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

    def load_frames_from_sprite_sheet(self, sheet_path, frame_width, frame_height, frame_count):
        sprite_sheet = pygame.image.load(sheet_path).convert_alpha()
        frames = []

        for i in range(frame_count):
            frame = sprite_sheet.subsurface(pygame.Rect(i * frame_width, 0, frame_width, frame_height))
            frames.append(frame)

        return frames

    def setup_animation_helpers(self):
        self.animation_timers = [0.5, 0.25, 0.25]
        self.animation_start_times = {"player": None, "opponent": None}
        self.current_animation_frames = {"player": 0, "opponent": 0}
        self.last_attack_names = {"player": None, "opponent": None}

    def play_music(self, key, fadeout_ms=1000, fadein_ms=1000):
        if self.current_music == key:
            return 

        if pygame.mixer.music.get_busy():
            pygame.mixer.music.fadeout(fadeout_ms)

        pygame.mixer.music.load(self.music_paths[key])
        pygame.mixer.music.play(-1, fade_ms=fadein_ms)
        self.current_music = key
    
    def load_attacks(self):
        self.kick_frames = self.load_frames_from_sprite_sheet(f"{ATTACK_FOLDER_PATH}/kick_sprite_sheet.png", 256, 256, 3)
        self.punch_frames = self.load_frames_from_sprite_sheet(f"{ATTACK_FOLDER_PATH}/punch_sprite_sheet.png", 256, 256, 3)
        self.ice_frames = self.load_frames_from_sprite_sheet(f"{ATTACK_FOLDER_PATH}/ice_sprite_sheet.png", 256, 256, 3)
        self.fire_frames = self.load_frames_from_sprite_sheet(f"{ATTACK_FOLDER_PATH}/fire_sprite_sheet.png", 256, 256, 3)
        self.lightning_frames = self.load_frames_from_sprite_sheet(f"{ATTACK_FOLDER_PATH}/lightning_sprite_sheet.png", 256, 256, 3)

        sound_paths = {
            "punch": f"{MUSIC_FOLDER_PATH}/punch.mp3",
            "kick": f"{MUSIC_FOLDER_PATH}/kick.mp3",
            "fire": f"{MUSIC_FOLDER_PATH}/fire.mp3",
            "ice": f"{MUSIC_FOLDER_PATH}/ice.mp3",
            "lightning": f"{MUSIC_FOLDER_PATH}/lightning.mp3",
        }

        for name, path in sound_paths.items():
            self.attack_sounds[name] = pygame.mixer.Sound(path)

    def play_attack_sound(self, attack_type):
        if attack_type not in self.attack_sounds:
            return

        sound = self.attack_sounds[attack_type]

        if attack_type not in self.attack_channels:
            self.attack_channels[attack_type] = pygame.mixer.find_channel()

        channel = self.attack_channels[attack_type]

        if channel and not channel.get_busy():
            pygame.mixer.music.set_volume(0.3)
            channel.play(sound)
        self.attack_restore_time = time.time() + sound.get_length()

    def initialize_game(self):
        self.load_attacks()
        self.setup_animation_helpers()
        if self.backend.isSinglePlayer or self.backend.isSinglePlayer == None:
            self.load_single_player_assets()
        else:
            self.load_multiplayer_assets()

        self.backend.start()

    def load_single_player_assets(self):
        self.running_bg = pygame.image.load(f"{ASSET_FOLDER_PATH}/background/gym.jpg")

        self.running_bg = pygame.transform.scale(
            self.running_bg, (SCREEN_WIDTH // 2, SCREEN_HEIGHT)
        )

        self.punching_bag = pygame.image.load(f"{OBJECTS_FOLDER_PATH}/punching_bag.png")

        punching_bag_height = int(SCREEN_HEIGHT * 0.6) 
        punching_bag_width = int(
            self.punching_bag.get_width()
            * (punching_bag_height / self.punching_bag.get_height())
        )

        self.punching_bag = pygame.transform.scale(
            self.punching_bag, (punching_bag_width, punching_bag_height)
        )

        self.punching_bag_rect = self.punching_bag.get_rect(
            center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2)
        )

    def load_multiplayer_assets(self):
        self.running_bg = pygame.image.load(f"{ASSET_FOLDER_PATH}/background/gym.jpg")

        self.running_bg = pygame.transform.scale(
            self.running_bg, (SCREEN_WIDTH, SCREEN_HEIGHT)
        )

        self.health_bar_width = 500
        self.health_bar_height = 50
        self.health_bar_border = 10

    def render_health_bars(self):
        self.player_health, self.opponent_health = (
            self.backend.multiplayer_client.get_health()
        )

        bar_x_offset = 50
        bar_y = 50

        def draw_health_bar(x, health_percent):

            health_width = int(self.health_bar_width * max(0, health_percent))
            filled_rect = pygame.Rect(x, bar_y, health_width, self.health_bar_height)
            background_rect = pygame.Rect(x, bar_y, self.health_bar_width, self.health_bar_height)

            outline_rect = background_rect.inflate(self.health_bar_border * 2, self.health_bar_border * 2)
            pygame.draw.rect(self.screen, WHITE, outline_rect)

            pygame.draw.rect(self.screen, RED, background_rect)
            pygame.draw.rect(self.screen, GREEN, filled_rect)

        player_health_percent = self.player_health / 100
        draw_health_bar(bar_x_offset, player_health_percent)

        opponent_health_percent = self.opponent_health / 100
        opponent_x = SCREEN_WIDTH - bar_x_offset - self.health_bar_width
        draw_health_bar(opponent_x, opponent_health_percent)


    def render_start(self):
        self.screen.blit(self.start_bg, (0, 0))
        self.screen.blit(self.title_img, self.title_rect)

        start_text_1 = self.info_font.render(
            "Say 'Single Player' or 'Multiplayer'", True, LILAC
        )
        start_rect_1 = start_text_1.get_rect(
            center=(SCREEN_WIDTH // 2, 50)
        )

        start_text_2 = self.info_font.render(
            "to start the game!", True, LILAC
        )
        start_rect_2 = start_text_2.get_rect(
            center=(SCREEN_WIDTH// 2, 90)
        )

        self.screen.blit(start_text_1, start_rect_1)
        self.screen.blit(start_text_2, start_rect_2)

    def render_initialize(self):
        self.screen.blit(self.inital_bg, (0, 0))
        self.screen.blit(self.initial_tutorial, self.initial_tutorial_rect)
        mode_type = 'Single Player' if (self.backend.isSinglePlayer or self.backend.isSinglePlayer == None) else 'Multiplayer'
        dot_states = [".", "..", "..."]
        elapsed = time.time() - self.init_start_time
        dot_index = int(elapsed * 2) % len(dot_states) 
        dots = dot_states[dot_index]

        init_text = self.info_font.render(f"Starting {mode_type} mode{dots}", True, LILAC)
        init_rect = init_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2))
        self.screen.blit(init_text, init_rect)

    def render_attack_animation(self, animation_type, x, y, mirrored=False, is_opponent=False):
        key = "opponent" if is_opponent else "player"
        current_time = time.time()

        timer = self.opponent_attack_timer if is_opponent else self.attack_timer

        if self.animation_start_times.get(key) is None or abs(current_time - timer) < 0.01:
            self.animation_start_times[key] = current_time
            self.current_animation_frames[key] = 0

        elapsed = current_time - self.animation_start_times[key]
        frame_durations = self.animation_timers

        total_duration = 0
        for i, duration in enumerate(frame_durations):
            total_duration += duration
            if elapsed < total_duration:
                self.current_animation_frames[key] = i
                break
        else:
            self.animation_start_times[key] = None
            self.current_animation_frames[key] = 0
            return

        frames = getattr(self, f"{animation_type}_frames", [])
        if frames and self.current_animation_frames[key] < len(frames):
            frame = frames[self.current_animation_frames[key]]
            if mirrored:
                frame = pygame.transform.flip(frame, True, False)
            self.screen.blit(frame, (x, y))


    def render_running(self, frame):
        if self.backend.isSinglePlayer or self.backend.isSinglePlayer == None:
            self.screen.blit(self.running_bg, (0, 0))
            self.screen.blit(self.punching_bag, self.punching_bag_rect)

            if frame is None:
                print("Error: No camera found")
                pygame.draw.rect(
                    self.screen, 
                    (0, 0, 0), 
                    pygame.Rect(SCREEN_WIDTH // 2, 0, SCREEN_WIDTH // 2, SCREEN_HEIGHT)
                )

                error_text = self.info_font.render("Error: No camera found", True, WHITE)
                error_rect = error_text.get_rect(center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT // 2))
                self.screen.blit(error_text, error_rect)
                
            else:  # CENTER CAMERA
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

            current_time = time.time()
            if self.attack and current_time < self.attack_timer:
                if self.attack == 'block':
                    attack_text = self.info_font.render("blocking...", True, WHITE)
                    attack_rect = attack_text.get_rect(center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4))
                    self.screen.blit(attack_text, attack_rect)
                else:
                    attack_text = self.info_font.render(self.attack, True, WHITE)
                    attack_rect = attack_text.get_rect(center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT * 3 // 4))
                    self.screen.blit(attack_text, attack_rect)

                    self.render_attack_animation(self.attack, SCREEN_WIDTH // 4 - 128, SCREEN_HEIGHT // 2 - 128)
                    self.play_attack_sound(self.attack)
        else:
            self.screen.blit(self.running_bg, (0, 0))

            if frame is not None:
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

                self.screen.blit(player_surface, (SCREEN_WIDTH // 2, 0))

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

                self.screen.blit(opponent_surface, (0, 0))

            self.render_health_bars()

            current_time = time.time()
            
            if self.attack and current_time < self.attack_timer:
                if self.attack == 'block':
                    attack_text = self.info_font.render("blocking...", True, WHITE)
                    attack_rect = attack_text.get_rect(center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4))
                    self.screen.blit(attack_text, attack_rect)
                else:
                    attack_text = self.info_font.render(self.attack, True, WHITE)
                    attack_rect = attack_text.get_rect(center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT * 3 // 4))
                    self.screen.blit(attack_text, attack_rect)

                    self.render_attack_animation(self.attack, SCREEN_WIDTH // 4 - 128, SCREEN_HEIGHT // 2 - 128)
                    self.play_attack_sound(self.attack)

            if self.opponent_attack and current_time < self.opponent_attack_timer:
                if self.opponent_attack == "block":
                    attack_text = self.info_font.render("blocking...", True, WHITE)
                    attack_rect = attack_text.get_rect(center=(SCREEN_WIDTH // 4, SCREEN_HEIGHT * 3 // 4))
                else:
                    opponent_text = self.info_font.render(self.opponent_attack, True, WHITE)
                    opponent_rect = opponent_text.get_rect(center=(SCREEN_WIDTH * 3 // 4, SCREEN_HEIGHT * 3 // 4))
                    self.screen.blit(opponent_text, opponent_rect)

                    self.render_attack_animation(
                        self.opponent_attack,
                        SCREEN_WIDTH * 3 // 4 - 128,
                        SCREEN_HEIGHT // 2 - 128,
                        mirrored=True,
                        is_opponent=True
                    )
    def render_end(self):
        self.screen.blit(self.start_bg, (0, 0))

        end_text_1 = self.info_font.render(
            "THANK YOU FOR PLAYING STREETFIGHTER-LIVE!", True, LILAC
        )
        end_rect_1 = end_text_1.get_rect(
            center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT * 0.8 // 2)
        )
        self.screen.blit(end_text_1, end_rect_1)

        if self.winner is not None:
            game_end_font = pygame.font.Font(
                    f"{FONTS_FOLDER_PATH}/PressStart2P-Regular.ttf", 50
                )
            message = "You are the winner! :)" if self.winner else "You lost :("
            end_text_2 = game_end_font.render(
                message, True, WHITE
            )
            end_rect_2 = end_text_2.get_rect(
                center=(SCREEN_WIDTH// 2, SCREEN_HEIGHT // 2)
            )
            self.screen.blit(end_text_2, end_rect_2)
       
        

    def render(self, frame):
        if self.state == START:
            self.render_start()
        elif self.state == INITIALIZE:
            self.render_initialize()
        elif self.state == RUNNING:
            self.render_running(frame)
        elif self.state == TERMINATE:
            self.render_end()

    def handle_events(self, input):
        current_time = time.time()

        if input != None:
            if self.state == START:
                self.state = INITIALIZE
                mode_type = 'SINGLE PLAYER' if input else 'MULTIPLAYER'
                print(f"STARTING {mode_type}")
                self.initialize_game()
                self.init_start_time = time.time()

            elif 'state' in input and input['state'] == 'terminate':
                self.state = TERMINATE
                self.winner = input['winner'] if 'winner' in input else None
                self.backend.terminate()

            elif self.state == RUNNING and "AttackType" in input:
                if self.attack is None or current_time >= self.attack_timer:
                    self.attack = input["AttackType"]
                    self.attack_timer = current_time + ATTACK_DISPLAY_DURATION

                if self.backend.isSinglePlayer == False:
                    if "player_health" in input:
                        self.player_health = input["player_health"]
                    if "opponent_health" in input:
                        self.opponent_health = input["opponent_health"]

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
                    self.initialize_game()
                    self.init_start_time = time.time()
                elif self.state == RUNNING and event.key == pygame.K_q:
                    self.backend.terminate()
                    self.state = TERMINATE

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
            
        if self.attack_restore_time and time.time() > self.attack_restore_time:
            pygame.mixer.music.set_volume(1.0)
            self.attack_restore_time = None
            
        if self.state in [START, INITIALIZE, TERMINATE]:
            self.play_music("general")
        elif self.state == RUNNING:
            if self.backend.isSinglePlayer:
                self.play_music("singleplayer")
            else:
                self.play_music("multiplayer")

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