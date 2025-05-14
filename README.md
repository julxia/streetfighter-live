# Streetfighter-LIVE

A real-time fighting game that combines voice commands and pose recognition to create an immersive gaming experience. Players can use physical movements and voice commands to control their character in both single-player and multiplayer modes.

## Table of Contents

### Core Modules

1. **Main Game (`main.py`)**

   - Controls the game flow and rendering
   - High level component that tntegrates all components together
   - Main entry point to the game

2. **Backend (`/backend`)**

   - `GameLogic.py`: Manages game state and logic and integrates recognition systems

3. **Recognition (`/recognition`)**

   - `voice.py`: Speech recognition for voice commands
   - `pose.py`: Manages pose detection interface
   - `models.py`: Integrates different recognition models
   - `model_types.py`: Defines common data structures
   - `/pose`: Contains pose recognition implementation
     - `pose_recognition.py`: Core camera interface that captures video feed, processes frames through MediaPipe's pose detection, and provides an API for the game to access detected poses and gestures.
     - `gesture_recognizer.py`: Analyzes spatial relationships between body landmarks to recognize specific fighting gestures (punches, kicks, blocks) with confidence scores.
     - `pose_callback.py`: Bridges MediaPipe's asynchronous pose detection results with the game system by processing raw landmark data into standardized gesture outputs.

4. **Multiplayer (`/multiplayer`)**

   - `game_server.py`: Handles multiplayer game sessions and networking
   - `network_client.py`: Client-side networking for multiplayer games

5. **Assets (`/assets`)**
   - `/music`: Game sound effects and background music
   - `/attacks`: Attack animation sprites
   - `/background`: Background images for different game states
   - `/fonts`: Game fonts
   - `/objects`: Game object images
   - `/title`: Title screen assets

## Setup and Installation

### Prerequisites

- Python 3.12 recommended
- Webcam for pose detection
- Microphone for voice commands
- Windows/MacOS

### Installation

1. Clone the repository:

   ```
   git clone https://github.com/julxia/streetfighter-live.git
   cd streetfighter-live
   ```

2. Create and activate a virtual environment:

   ```
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:

   ```
   # Windows
   pip install -r requirements.txt

   # macOS
   pip install -r requirements-mac.txt
   ```

4. Create a `.env` file in the root directory with the following content:
   ```
   # For single-player mode, this can be any value
   # For multiplayer mode, set this to the server's IPv4 address
   IPV4_ADDR=your_server_ip_address
   ```

### Running the Game

#### Single Player Mode

1. Start the game:

   ```
   python main.py
   ```

2. Say "Single Player" when prompted to start a single-player game.

#### Multiplayer Mode

**Server Setup (One Player - Host)**:

1. Set up port forwarding on port 5555 on your router if playing over the internet
2. For Windows, allow inbound connections for port 5555 in Windows Firewall
3. Find your IPv4 address:

   ```
   # Windows
   ipconfig

   # macOS/Linux
   ifconfig
   ```

4. Update the `.env` file with your IPv4 address
5. Start the multiplayer server:
   ```
   python ./multiplayer/game_server.py
   ```

**Client Setup (All Players)**:

1. Update the `.env` file with the server's IPv4 address
2. Start the game:
   ```
   python main.py
   ```
3. Say "Multiplayer" when prompted to connect to the server

## Gameplay

### Controls

- **Voice Commands**:

  - "Single Player" / "Multiplayer": Start game mode
  - "Exit": Exit the current game
  - Special Moves: "Fire", "Ice", "Lightning"

- **Pose Recognition**:
  - Punch: Extend one arm forward, while keeping the other one bent in a blocked position
  - Kick: Lift leg up, making sure the ankle is above the knee which is above the hip
  - Block: Bend both arms in front of your shoulders and keep your fist next to your chin

### Game Modes

- **Single Player**: Practice mode with a punching bag
- **Multiplayer**: Play against another player over the local network

## Troubleshooting

- **Camera not detected**: Ensure your webcam is connected and not in use by another application. For Mac, make sure the camera isn't using the one on your iPhone.
- **Microphone not working**: Check your microphone settings and ensure it has proper permissions
- **Multiplayer connection issues**:
  - Verify both the server and client `.env` files have the correct IPv4 address
  - Ensure port 5555 is open on the server's firewall
  - Check network connectivity between client and server
  - Start server and then have the clients join

## Development

- The project uses pygame for rendering and game logic
- OpenCV and MediaPipe are used for pose recognition
- Speech recognition is handled by the SpeechRecognition library

## System Requirements

- **OS**: Windows 11, or MacOS
- **Processor**: Modern dual-core CPU or better
- **RAM**: 4GB or more
- **GPU**: Basic integrated graphics or better
- **Webcam and Microphone**: Required for gameplay
