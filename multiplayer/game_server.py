import socket
import threading
import pickle
import time
import numpy as np
import struct
import signal
import sys
import atexit


class GameServer:
    def __init__(self, host="0.0.0.0", port=5555):
        self.host = host
        self.port = port
        self.server = None
        self.clients = []
        self.client_addresses = []
        self.client_frames = [None, None]
        self.client_actions = [None, None]
        self.player_health = [100, 100]
        self.game_running = False
        self.running = False
        self.threads = []
        self.next_client_id = 0  # Track the next available client ID
        self.client_ids = {}  # Map client sockets to their IDs
        self.defender_is_blocking = False

        # Register cleanup handlers
        atexit.register(self.cleanup)
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

        print(f"Server initialized. Will start on {host}:{port}")

    def setup_socket(self):
        """Set up the server socket with proper options"""
        # Create a new socket
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Set socket options for proper cleanup
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # On some systems, also use SO_REUSEPORT if available
        if hasattr(socket, "SO_REUSEPORT"):
            self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)

        # Set timeout to allow checking for shutdown
        self.server.settimeout(1.0)

        try:
            self.server.bind((self.host, self.port))
            self.server.listen(2)
            print(f"Server started successfully on {self.host}:{self.port}")
            return True
        except OSError as e:
            print(f"Error binding to port {self.port}: {e}")
            self.server.close()
            return False

    def handle_client(self, client, client_id):
        """Handle communication with a connected client"""
        print(f"Client {client_id} connected")

        try:
            while self.running:
                try:
                    # First receive the message size
                    size_data = self.recv_all(client, 4)
                    if not size_data:
                        break

                    msg_size = struct.unpack("!I", size_data)[0]

                    # Now receive the actual message
                    data = self.recv_all(client, msg_size)
                    if not data:
                        break

                    # Unpack the data
                    client_data = pickle.loads(data)

                    # Update client data - using client_id as index safely
                    if "action" in client_data:
                        action = client_data["action"]

                        # Check if this client has a valid opponent
                        has_opponent = False
                        for potential_opponent in self.clients:
                            if (
                                potential_opponent != client
                                and potential_opponent in self.client_ids
                            ):
                                opponent_id = self.client_ids[potential_opponent]
                                has_opponent = True
                                break

                        # Safely update the client action
                        while len(self.client_actions) <= client_id:
                            self.client_actions.append(None)
                        self.client_actions[client_id] = action

                        # Process attack if valid and game is running with an opponent
                        if (
                            has_opponent
                            and action
                            and isinstance(action, dict)
                            and "AttackType" in action
                        ):
                            self.process_attack(client_id, action["AttackType"])

                    if "frame" in client_data:
                        compressed_frame = client_data["frame"]
                        if compressed_frame is not None:
                            # Safely update the client frame
                            while len(self.client_frames) <= client_id:
                                self.client_frames.append(None)
                            self.client_frames[client_id] = compressed_frame

                    # Determine opponent ID
                    opponent_id = None
                    opponent_frame = None
                    opponent_action = None
                    opponent_health = 100

                    # Find an opponent's data if one exists
                    for potential_opponent in self.clients:
                        if (
                            potential_opponent != client
                            and potential_opponent in self.client_ids
                        ):
                            opponent_id = self.client_ids[potential_opponent]
                            if opponent_id < len(self.client_frames):
                                opponent_frame = self.client_frames[opponent_id]
                            if opponent_id < len(self.client_actions):
                                opponent_action = self.client_actions[opponent_id]
                            if opponent_id < len(self.player_health):
                                opponent_health = self.player_health[opponent_id]
                            break

                    # Get player health safely
                    player_health = 100
                    if client_id < len(self.player_health):
                        player_health = self.player_health[client_id]

                    # Send game state back to client
                    game_state = {
                        "opponent_frame": opponent_frame,
                        "opponent_action": opponent_action,
                        "player_health": player_health,
                        "opponent_health": opponent_health,
                        "game_running": self.game_running,
                        "players_connected": len(self.clients),
                    }

                    # Serialize the response
                    response = pickle.dumps(game_state)

                    # Send size of message first, then the message
                    client.sendall(struct.pack("!I", len(response)))
                    client.sendall(response)

                except socket.timeout:
                    # Just a timeout, continue
                    continue
                except (ConnectionResetError, BrokenPipeError) as e:
                    print(f"Connection with client {client_id} lost: {e}")
                    break

        except Exception as e:
            print(f"Error handling client {client_id}: {e}")
        finally:
            print(f"Client {client_id} disconnected")
            self.disconnect_client(client)

    def disconnect_client(self, client):
        """Safely disconnect a client"""
        if client in self.clients:
            self.clients.remove(client)

            # Find and remove client ID
            if client in self.client_ids:
                client_id = self.client_ids.pop(client)
                print(f"Removed client_id {client_id}")

                # Clear associated client data
                if client_id < len(self.client_frames):
                    self.client_frames[client_id] = None
                if client_id < len(self.client_actions):
                    self.client_actions[client_id] = None

        try:
            client.close()
        except:
            pass

        if len(self.clients) < 2:
            self.game_running = False

    def recv_all(self, sock, n):
        """Helper function to receive n bytes or return None if EOF is hit"""
        sock.settimeout(0.5)  # Short timeout to allow checking for shutdown

        data = bytearray()
        start_time = time.time()
        while len(data) < n and self.running:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)

                # Check for timeout on large transfers
                if time.time() - start_time > 5.0:  # 5 second max for any receive
                    print("Receive operation timed out")
                    return None

            except socket.timeout:
                # Just a timeout, continue if we're still running
                if not self.running:
                    return None
                continue
            except socket.error:
                return None

        return data if len(data) == n else None

    def process_attack(self, attacker_id, attack_type):
        """Process an attack from a player"""
        if not self.game_running:
            return

        # Find opponent
        defender_id = None
        for client, id in self.client_ids.items():
            if id != attacker_id:
                defender_id = id
                break

        if defender_id is None:
            return  # No opponent found

        # Ensure player health arrays are large enough
        while len(self.player_health) <= max(attacker_id, defender_id):
            self.player_health.append(100)

        # Check if defender is blocking
        if defender_id < len(self.client_actions) and self.client_actions[defender_id]:
            defender_action = self.client_actions[defender_id]
            if (
                isinstance(defender_action, dict)
                and defender_action.get("AttackType") == "block"
            ):
                self.defender_is_blocking = True
            elif (
                isinstance(defender_action, dict)
                and defender_action.get("AttackType") != "block"
            ):
                self.defender_is_blocking = False

        # Define damage for different attack types
        damage = 0
        if attack_type and isinstance(attack_type, str):
            attack_lower = attack_type.lower()

            # If defender is blocking, check attack type
            if self.defender_is_blocking:
                # Block completely negates punch and kick damage
                if attack_lower in {
                    "punch",
                    "kick",
                }:
                    damage = 0  # No damage when blocking punches or kicks
                else:
                    # Special moves still do damage even when blocking (reduced)
                    if attack_lower in {"lightning", "fire", "ice"}:
                        damage = 3  # Reduced damage for special moves
            else:
                # Normal damage when not blocking
                if attack_lower == "punch":
                    damage = 5
                elif attack_lower == "kick":
                    damage = 7
                elif attack_lower in {"lightning", "fire", "ice"}:
                    damage = 10
                elif attack_lower == "block":
                    damage = 0

        # Apply damage to defender
        self.player_health[defender_id] -= damage

        # Cap health at 0
        if self.player_health[defender_id] <= 0:
            self.player_health[defender_id] = 0
            self.game_running = False

    def start_game(self):
        """Start the game when two players are connected"""
        if len(self.clients) == 2 and not self.game_running:
            print("Starting game with 2 players")
            self.game_running = True

            # Reset health for all active players
            client_ids = list(self.client_ids.values())
            max_id = max(client_ids) if client_ids else 0

            # Resize and reset health array
            self.player_health = [100] * (max_id + 1)

    def run(self):
        """Main server loop to accept clients"""
        if not self.setup_socket():
            print("Failed to set up server socket. Exiting.")
            return

        self.running = True

        print("Server is running. Press Ctrl+C to stop.")

        try:
            while self.running:
                try:
                    client, addr = self.server.accept()

                    if len(self.clients) < 2:
                        # Assign the next available client ID
                        client_id = self.next_client_id
                        self.next_client_id += 1

                        self.clients.append(client)
                        self.client_ids[client] = client_id
                        self.client_addresses.append(addr)

                        # Make sure our data arrays are large enough
                        while len(self.client_frames) <= client_id:
                            self.client_frames.append(None)
                        while len(self.client_actions) <= client_id:
                            self.client_actions.append(None)
                        while len(self.player_health) <= client_id:
                            self.player_health.append(100)

                        # Start a thread to handle this client
                        thread = threading.Thread(
                            target=self.handle_client, args=(client, client_id)
                        )
                        thread.daemon = True
                        thread.start()
                        self.threads.append(thread)

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
                except socket.timeout:
                    # This is expected due to the socket timeout we set
                    continue
                except Exception as e:
                    if self.running:  # Only log if not shutting down
                        print(f"Error in main server loop: {e}")

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received")
        finally:
            self.cleanup()

    def signal_handler(self, sig, frame):
        """Handle termination signals gracefully"""
        print(f"\nSignal {sig} received, shutting down...")
        self.cleanup()
        sys.exit(0)

    def cleanup(self):
        """Clean up all resources"""
        print("Cleaning up server resources...")
        self.running = False

        # Close all client connections
        clients_copy = (
            self.clients.copy()
        )  # Create a copy to avoid modification during iteration
        for client in clients_copy:
            self.disconnect_client(client)

        # Close the server socket
        if hasattr(self, "server") and self.server:
            try:
                self.server.close()
                print("Server socket closed")
            except:
                pass

        # Wait for threads to finish
        for thread in self.threads:
            if thread.is_alive():
                thread.join(timeout=1.0)

        print("Server shutdown complete")


if __name__ == "__main__":
    server = GameServer()
    server.run()
