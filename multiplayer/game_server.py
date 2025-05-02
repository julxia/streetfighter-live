import socket
import threading
import pickle
import time
import numpy as np
import struct


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
                # First receive the message size (4 bytes for a 32-bit unsigned int)
                size_data = self.recv_all(client, 4)
                if not size_data:
                    break

                msg_size = struct.unpack("!I", size_data)[0]

                # Now receive the actual message
                data = self.recv_all(client, msg_size)
                if not data:
                    break

                # Unpack the data (action and compressed frame)
                client_data = pickle.loads(data)

                # Update client data
                if "action" in client_data:
                    action = client_data["action"]
                    self.client_actions[client_id] = action

                    # Process attack if valid
                    if action and isinstance(action, dict) and "AttackType" in action:
                        self.process_attack(client_id, action["AttackType"])

                if "frame" in client_data:
                    compressed_frame = client_data["frame"]
                    if compressed_frame is not None:
                        # Store compressed frame
                        self.client_frames[client_id] = compressed_frame

                # Send game state back to client
                game_state = {
                    "opponent_frame": (
                        self.client_frames[1 - client_id]
                        if len(self.clients) > 1
                        else None
                    ),
                    "opponent_action": (
                        self.client_actions[1 - client_id]
                        if len(self.clients) > 1
                        else None
                    ),
                    "player_health": self.player_health[client_id],
                    "opponent_health": (
                        self.player_health[1 - client_id]
                        if len(self.clients) > 1
                        else 100
                    ),
                    "game_running": self.game_running,
                    "players_connected": len(self.clients),
                }

                # Serialize the response
                response = pickle.dumps(game_state)

                # Send size of message first, then the message
                client.sendall(struct.pack("!I", len(response)))
                client.sendall(response)

        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        finally:
            print(f"Client {client_id} disconnected")
            if client in self.clients:
                self.clients.remove(client)

            if len(self.clients) < 2:
                self.game_running = False
            client.close()

    def recv_all(self, sock, n):
        """Helper function to receive n bytes or return None if EOF is hit"""
        data = bytearray()
        while len(data) < n:
            packet = sock.recv(n - len(data))
            if not packet:
                return None
            data.extend(packet)
        return data

    def process_attack(self, attacker_id, attack_type):
        """Process an attack from a player"""
        if not self.game_running:
            return

        defender_id = 1 - attacker_id  # The other player

        # Define damage for different attack types
        damage = 0
        if attack_type and isinstance(attack_type, str):
            attack_lower = attack_type.lower()
            if attack_lower == "punch":
                damage = 5
            elif attack_lower == "kick":
                damage = 7
            elif attack_lower == "block":
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
                    # Server is full - send an error message
                    error_msg = pickle.dumps({"error": "Server full"})
                    try:
                        client.sendall(struct.pack("!I", len(error_msg)))
                        client.sendall(error_msg)
                    except:
                        pass
                    client.close()
        except KeyboardInterrupt:
            print("Server shutting down...")
        finally:
            if hasattr(self, "server"):
                self.server.close()


if __name__ == "__main__":
    server = GameServer()
    server.run()
