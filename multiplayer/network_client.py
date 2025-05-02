import socket
import pickle
import cv2
import numpy as np
import threading
import time
import struct


class NetworkClient:
    def __init__(self, host="localhost", port=5555):
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server = host
        self.port = port
        self.addr = (self.server, self.port)
        self.connected = False
        self.opponent_frame = None
        self.opponent_action = None
        self.player_health = 100
        self.opponent_health = 100
        self.game_running = False
        self.players_connected = 0
        self.lock = threading.Lock()

    def connect(self):
        """Connect to the server"""
        try:
            self.client.connect(self.addr)
            self.connected = True

            # Start receiving thread
            self.receive_thread = threading.Thread(target=self.receive_data)
            self.receive_thread.daemon = True
            self.receive_thread.start()

            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def send_data(self, action, frame):
        """Send action and frame data to server"""
        if not self.connected:
            return

        try:
            # Compress the frame if it exists
            compressed_frame = None
            if frame is not None:
                # Resize to smaller dimensions for network efficiency
                resized_frame = cv2.resize(frame, (320, 240))
                # Convert to JPEG format for compression
                _, compressed_frame = cv2.imencode(
                    ".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, 40]
                )

            # Create data packet
            data = {"action": action, "frame": compressed_frame}

            # Serialize the data
            serialized_data = pickle.dumps(data)

            # Send size of message first, then the message
            self.client.sendall(struct.pack("!I", len(serialized_data)))
            self.client.sendall(serialized_data)

        except Exception as e:
            print(f"Error sending data: {e}")
            self.connected = False

    def receive_data(self):
        """Background thread to receive data from server"""
        while self.connected:
            try:
                # First receive the message size
                size_data = self.recv_all(self.client, 4)
                if not size_data:
                    break

                msg_size = struct.unpack("!I", size_data)[0]

                # Now receive the actual message
                data = self.recv_all(self.client, msg_size)
                if not data:
                    break

                game_state = pickle.loads(data)

                with self.lock:
                    # Handle opponent frame if available
                    if (
                        "opponent_frame" in game_state
                        and game_state["opponent_frame"] is not None
                    ):
                        # Decompress JPEG frame
                        compressed_frame = game_state["opponent_frame"]
                        opponent_frame_array = np.frombuffer(
                            compressed_frame, dtype=np.uint8
                        )
                        self.opponent_frame = cv2.imdecode(
                            opponent_frame_array, cv2.IMREAD_COLOR
                        )

                    # Update other game state
                    if "opponent_action" in game_state:
                        self.opponent_action = game_state["opponent_action"]

                    if "player_health" in game_state:
                        self.player_health = game_state["player_health"]

                    if "opponent_health" in game_state:
                        self.opponent_health = game_state["opponent_health"]

                    if "game_running" in game_state:
                        self.game_running = game_state["game_running"]

                    if "players_connected" in game_state:
                        self.players_connected = game_state["players_connected"]

                    if "error" in game_state:
                        print(f"Server error: {game_state['error']}")
                        self.connected = False

            except Exception as e:
                print(f"Error receiving data: {e}")
                self.connected = False
                break

    def recv_all(self, sock, n):
        """Helper function to receive n bytes or return None if EOF is hit"""
        data = bytearray()
        while len(data) < n:
            try:
                packet = sock.recv(n - len(data))
                if not packet:
                    return None
                data.extend(packet)
            except socket.error as e:
                print(f"Socket error: {e}")
                return None
        return data

    def get_opponent_frame(self):
        """Get the most recent opponent frame"""
        with self.lock:
            return self.opponent_frame

    def get_opponent_action(self):
        """Get the most recent opponent action"""
        with self.lock:
            action = self.opponent_action
            # Reset after reading to avoid repeating the same action
            self.opponent_action = None
            return action

    def get_health(self):
        """Get current health values"""
        with self.lock:
            return self.player_health, self.opponent_health

    def is_game_running(self):
        """Check if game is running on server"""
        with self.lock:
            return self.game_running

    def get_players_connected(self):
        """Get the number of players connected to the server"""
        with self.lock:
            return self.players_connected

    def disconnect(self):
        """Disconnect from the server"""
        self.connected = False
        if hasattr(self, "client"):
            self.client.close()
