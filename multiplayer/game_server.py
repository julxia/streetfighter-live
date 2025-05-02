import socket
import threading
import pickle
import time
import numpy as np


class GameServer:
    def __init__(self, host="0.0.0.0", port=5555):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(2)  # Listen for 2 clients max

        self.clients = []
        self.client_addresses = []
        self.client_frames = [None, None]
        self.client_actions = [None, None]
        self.player_health = [100, 100]  # Health for each player
        self.game_running = False

        print(f"Server started on {host}:{port}")

    def handle_client(self, client, client_id):
        """Handle communication with a connected client"""
        print(f"Client {client_id} connected")

        try:
            while True:
                # Receive data from client
                try:
                    data = client.recv(4096)
                    if not data:
                        break

                    # Unpack the data (action and compressed frame)
                    client_data = pickle.loads(data)

                    # Update client data
                    if "action" in client_data:
                        action = client_data["action"]
                        self.client_actions[client_id] = action

                        # Process attack if valid
                        if action and "AttackType" in action:
                            self.process_attack(client_id, action["AttackType"])

                    if "frame" in client_data:
                        compressed_frame = client_data["frame"]
                        if compressed_frame is not None:
                            # Store compressed frame
                            self.client_frames[client_id] = compressed_frame

                    # Send game state back to client
                    game_state = {
                        "opponent_frame": self.client_frames[1 - client_id],
                        "opponent_action": self.client_actions[1 - client_id],
                        "player_health": self.player_health[client_id],
                        "opponent_health": self.player_health[1 - client_id],
                        "game_running": self.game_running,
                    }

                    client.sendall(pickle.dumps(game_state))
                except socket.error:
                    break

        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        finally:
            print(f"Client {client_id} disconnected")
            if client in self.clients:
                self.clients.remove(client)

            if len(self.clients) < 2:
                self.game_running = False
            client.close()

    def process_attack(self, attacker_id, attack_type):
        """Process an attack from a player"""
        if not self.game_running:
            return

        defender_id = 1 - attacker_id  # The other player

        # Define damage for different attack types
        damage = 0
        if attack_type.lower() == "punch":
            damage = 5
        elif attack_type.lower() == "kick":
            damage = 7
        elif attack_type.lower() == "block":
            return  # No damage for blocks

        # Apply damage to defender
        self.player_health[defender_id] -= damage

        # Cap health at 0
        if self.player_health[defender_id] < 0:
            self.player_health[defender_id] = 0

    def start_game(self):
        """Start the game when two players are connected"""
        if len(self.clients) == 2 and not self.game_running:
            print("Starting game with 2 players")
            self.game_running = True
            self.player_health = [100, 100]  # Reset health

    def run(self):
        """Main server loop to accept clients"""
        client_id = 0

        try:
            while True:
                client, addr = self.server.accept()

                if len(self.clients) < 2:
                    self.clients.append(client)
                    self.client_addresses.append(addr)

                    # Start a thread to handle this client
                    thread = threading.Thread(
                        target=self.handle_client, args=(client, client_id)
                    )
                    thread.daemon = True
                    thread.start()

                    client_id += 1

                    # Start the game if we have two players
                    if len(self.clients) == 2:
                        self.start_game()
                else:
                    # Server is full
                    client.sendall(pickle.dumps({"error": "Server full"}))
                    client.close()
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            self.server.close()


if __name__ == "__main__":
    server = GameServer()
    server.run()
