# AI-Based-Image-Grid-Game-with-Leaderboards-and-Smart-Swapping
This repository contains an interactive Python-based image puzzle game built with matplotlib, powered by a CNN-based tile classifier, and enhanced with user-friendly features like undo/redo, hints, restarts, and a leaderboard system categorized by grid size.

🎯 Project Objective
An intelligent puzzle-solving system that allows users to upload any image, play a shuffled tile puzzle, track performance, and optionally auto-solve using a CNN model trained to classify piece positions.

🚀 Features
🧠 CNN-Based Solver
CNN model classifies each tile to its original position.
Users can view model predictions or manually solve the puzzle.

🧩 Interactive Puzzle Interface
Built with matplotlib for visual interaction.
Swap tiles using mouse clicks.
Real-time move counter and timer.

♻️ Game Controls
Undo/Redo: Reverse or reapply moves (u/y keys).
Restart: Reset the puzzle to initial state (r key).
Hint: Display original image (limit of 3 hints per game, press h).
Quit: Exit the game (q key).

🏆 Leaderboard
Stores top 10 scores per grid size (2x2, 3x3, 4x4, 5x5).
Sorts by fewest moves and fastest completion time.
View leaderboard in:
📊 Popup window (l key)
📋 Terminal (t key)

🖼️ Example Gameplay (Controls)

Controls:
- Click to select and swap tiles
- 'r' : Restart puzzle
- 'u' : Undo
- 'y' : Redo
- 'h' : Hint (max 3)
- 'l' : Leaderboard (window)
- 't' : Leaderboard (terminal)
- 'q' : Quit
  
🧱 System Architecture
Frontend: matplotlib GUI for gameplay and event handling.
CNN Model: Keras-based model trained on puzzle tile data.
Leaderboard: JSON-based storage grouped by grid size.
Input Handling: argparse CLI interface for loading leaderboard or launching gameplay.

🛠 Installation & Setup
✅ Prerequisites
Python 3.8+
pip (Python package manager)
Virtual environment (recommended)

📦 Installation
git clone https://github.com/srijairugu/ai-puzzle-game.git
cd ai-puzzle-game

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
▶️ Running the Game
🎮 Start Puzzle Game
python main.py
You will be prompted to enter:
Your name:srija
Path to the image:lion.jpg
Desired grid size (e.g., 3 3 for 3x3 grid):3 3

📈 Show Leaderboard only
python main.py --leaderboard
📊 Leaderboard Example
=== LEADERBOARD (3x3) ===
Rank | Time     | Name     | Moves
----------------------------------------
 1🥇 | 1:42.35  | srija    |    15
 2🥈 | 2:10.12  | Bob      |    16
 3🥉 | 2:35.88  | Carol    |    17
 
🧪 Model Details
CNN Architecture
Convolutional layers (32, 64, 128 filters)
Batch normalization and dropout
Dense layer with softmax classification

Input
Normalized image tiles of shape (64, 64, 3)
Labels representing tile positions

Training 
You can train the CNN model separately using synthetic shuffled puzzle pieces and supervised learning with cross-entropy loss.

📁 Project Structure
├── puzzle_solver.py               # Main puzzle game code
├── leaderboard.json               # Persistent leaderboard data
├── image.jpg
├── lion.jpg
├── requirements.txt
└── README.md

👨‍💻 Developer Info
Project by srija irugu
CNN-based solving, interactive UI, undo/redo and leaderboard fully integrated

📚 References
Matplotlib for GUI
Keras/TensorFlow for CNN
PIL & skimage for image handling

