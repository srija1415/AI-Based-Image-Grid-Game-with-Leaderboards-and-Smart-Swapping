import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import json
import tensorflow as tf
from tensorflow.keras import layers, models, utils
from skimage.transform import resize
import logging
import argparse  # Add argparse for command line arguments

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Leaderboard file
LEADERBOARD_FILE = "leaderboard.json"

# Load and save leaderboard
def load_leaderboard():
    """Load the leaderboard data from the JSON file with sections for different grid sizes."""
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as file:
            try:
                return json.load(file)
            except json.JSONDecodeError:
                return {"2x2": [], "3x3": [], "4x4": [], "5x5": []}
    return {"2x2": [], "3x3": [], "4x4": [], "5x5": []}

def save_leaderboard(leaderboard):
    """Save the leaderboard data with sections for different grid sizes."""
    with open(LEADERBOARD_FILE, "w") as file:
        json.dump(leaderboard, file, indent=4)

def update_leaderboard(username, moves, time_taken, grid_size):
    """Update the leaderboard for the specific grid size."""
    leaderboard = load_leaderboard()
    grid_key = f"{grid_size[0]}x{grid_size[1]}"
    
    # Create section if it doesn't exist
    if grid_key not in leaderboard:
        leaderboard[grid_key] = []
    
    # Add new score to the appropriate section
    leaderboard[grid_key].append({
        "name": username,
        "moves": moves,
        "time": time_taken
    })
    
    # Sort the specific grid size section
    leaderboard[grid_key].sort(key=lambda x: (x["moves"], x["time"]))
    
    # Keep only top 10 scores for each grid size
    leaderboard[grid_key] = leaderboard[grid_key][:10]
    
    save_leaderboard(leaderboard)

def split_image(image_path, grid_size=(3, 3)):
    try:
        # Load and preprocess image
        image = Image.open(image_path)
        target_size = (grid_size[1] * 64, grid_size[0] * 64)
        
        # Store original image for hints
        global original_image
        original_image = np.array(image)
        original_image = resize(original_image, target_size, anti_aliasing=True)
        
        # Convert to numpy array and normalize
        image_array = np.array(image)
        image_array = resize(image_array, target_size, anti_aliasing=True)
        
        # Calculate piece dimensions
        piece_height = image_array.shape[0] // grid_size[0]
        piece_width = image_array.shape[1] // grid_size[1]
        
        # Split image into pieces
        pieces = []
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                piece = image_array[i * piece_height:(i + 1) * piece_height,
                                  j * piece_width:(j + 1) * piece_width]
                pieces.append(piece)
        
        # Convert to numpy array
        pieces = np.array(pieces)
        
        # Shuffle pieces
        indices = np.arange(len(pieces))
        np.random.shuffle(indices)
        shuffled_pieces = pieces[indices]
        
        return shuffled_pieces, indices
            
    except Exception as e:
        logger.error(f"Error in split_image: {e}")
        raise

def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2), padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def solve_puzzle(model, pieces):
    try:
        predictions = model.predict(pieces, batch_size=32)
        sorted_indices = np.argsort(np.argmax(predictions, axis=1))
        return pieces[sorted_indices]
    except Exception as e:
        logger.error(f"Error in solve_puzzle: {e}")
        raise

# Global variables
selected_piece = None
moves = 0
pieces = []
grid_size = (3, 3)
current_order = None
solved = False
fig = None
axes = None
original_image = None  # Store the original image
hint_fig = None  # Store the hint figure
hint_usage_count = 0  # Track number of times hint was used
HINT_LIMIT = 3  # Fixed hint limit
# Add history tracking
move_history = []  # List to store past states
redo_history = []  # List to store undone states
start_state = None  # Store initial state for restart

def save_state():
    """Save current state to history."""
    global pieces, current_order, move_history, redo_history
    try:
        # Clear redo history when making a new move
        redo_history = []
        # Save current state
        move_history.append({
            'pieces': pieces.copy(),
            'current_order': current_order.copy(),
            'moves': moves
        })
        logger.info(f"State saved. Total states: {len(move_history)}")
    except Exception as e:
        logger.error(f"Error in save_state: {e}")

def undo_move():
    """Undo the last move."""
    global pieces, current_order, moves, move_history, redo_history, selected_piece
    try:
        if move_history:
            # Save current state to redo history
            redo_history.append({
                'pieces': pieces.copy(),
                'current_order': current_order.copy(),
                'moves': moves
            })
            # Restore previous state
            prev_state = move_history.pop()
            pieces = prev_state['pieces'].copy()
            current_order = prev_state['current_order'].copy()
            moves = prev_state['moves']
            selected_piece = None
            logger.info(f"Undo performed. Moves: {moves}")
            display_puzzle()
            plt.draw()
    except Exception as e:
        logger.error(f"Error in undo_move: {e}")

def redo_move():
    """Redo the last undone move."""
    global pieces, current_order, moves, move_history, redo_history, selected_piece
    try:
        if redo_history:
            # Save current state to move history
            move_history.append({
                'pieces': pieces.copy(),
                'current_order': current_order.copy(),
                'moves': moves
            })
            # Restore next state
            next_state = redo_history.pop()
            pieces = next_state['pieces'].copy()
            current_order = next_state['current_order'].copy()
            moves = next_state['moves']
            selected_piece = None
            logger.info(f"Redo performed. Moves: {moves}")
            display_puzzle()
            plt.draw()
    except Exception as e:
        logger.error(f"Error in redo_move: {e}")

def restart_puzzle():
    """Restart the puzzle from the beginning."""
    global pieces, current_order, moves, move_history, redo_history, solved, hint_usage_count
    if start_state is not None:
        pieces = start_state['pieces'].copy()
        current_order = start_state['current_order'].copy()
        moves = 0
        move_history = []
        redo_history = []
        solved = False
        hint_usage_count = 0  # Reset hint usage count
        display_puzzle()

def display_puzzle():
    global fig, axes, pieces, grid_size
    try:
        if fig is None:
            fig = plt.figure(figsize=(5, 5))
        else:
            fig.clear()
        
        axes = fig.subplots(grid_size[0], grid_size[1])
        axes = np.array(axes)

        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                idx = i * grid_size[1] + j
                axes[i, j].imshow(pieces[idx])
                axes[i, j].axis('off')

        plt.suptitle(f"Moves: {moves}")
        plt.draw()
    except Exception as e:
        logger.error(f"Error in display_puzzle: {e}")
        raise

def on_click(event):
    global selected_piece, pieces, moves, grid_size, current_order, solved

    if not event.inaxes:
        return

    try:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                ax = axes[i, j]
                if ax == event.inaxes:
                    clicked_index = i * grid_size[1] + j

                    if selected_piece is None:
                        selected_piece = clicked_index
                    else:
                        # Save state before making move
                        save_state()
                        # Swap pieces
                        pieces[[selected_piece, clicked_index]] = pieces[[clicked_index, selected_piece]]
                        current_order[[selected_piece, clicked_index]] = current_order[[clicked_index, selected_piece]]
                        selected_piece = None
                        moves += 1

                        display_puzzle()

                        if np.array_equal(current_order, np.arange(len(pieces))):
                            solved = True
                            logger.info("Puzzle solved!")
                            plt.close()
                    return
    except Exception as e:
        logger.error(f"Error in on_click: {e}")
        raise

def show_hint():
    """Display the original image as a hint."""
    global hint_fig, hint_usage_count
    try:
        if hint_usage_count >= HINT_LIMIT:
            # Show message in a new figure
            msg_fig = plt.figure(figsize=(4, 2))
            plt.text(0.5, 0.5, "Hint limit reached!",
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=plt.gca().transAxes)
            plt.axis('off')
            plt.draw()
            plt.show(block=False)
            # Close message after 2 seconds
            plt.pause(2)
            plt.close(msg_fig)
            return

        # Always create a new figure to ensure proper event handling
        if hint_fig is not None:
            plt.close(hint_fig)
        
        hint_fig = plt.figure(figsize=(5, 5))
        plt.imshow(original_image)
        plt.title(f"Hint - Original Image ({hint_usage_count + 1}/{HINT_LIMIT})")
        plt.axis('off')
        
        # Connect the escape key event to the hint figure
        hint_fig.canvas.mpl_connect('key_press_event', lambda event: hide_hint() if event.key == 'escape' else None)
        
        plt.draw()
        plt.show(block=False)
        hint_usage_count += 1
        logger.info(f"Hint displayed ({hint_usage_count}/{HINT_LIMIT})")
    except Exception as e:
        logger.error(f"Error in show_hint: {e}")
        hint_fig = None

def hide_hint():
    """Hide the hint window."""
    global hint_fig
    try:
        if hint_fig is not None:
            plt.close(hint_fig)
            hint_fig = None
            logger.info("Hint hidden")
    except Exception as e:
        logger.error(f"Error in hide_hint: {e}")
        hint_fig = None

def display_leaderboard_window():
    """Display the leaderboard in a new window."""
    try:
        leaderboard = load_leaderboard()
        
        # Create a new figure for the leaderboard
        lb_fig = plt.figure(figsize=(8, 6))
        plt.axis('off')
        
        # Create text content for the leaderboard
        text_content = "=== LEADERBOARD ===\n\n"
        
        for grid_key in sorted(leaderboard.keys()):
            if leaderboard[grid_key]:  # Only show sections with scores
                text_content += f"\n{grid_key} Grid:\n"
                text_content += "Rank | Time     | Name     | Moves\n"
                text_content += "-" * 40 + "\n"
                
                for i, entry in enumerate(leaderboard[grid_key], 1):
                    # Add special emoji for top 3
                    rank_emoji = ""
                    if i == 1:
                        rank_emoji = "🥇"
                    elif i == 2:
                        rank_emoji = "🥈"
                    elif i == 3:
                        rank_emoji = "🥉"
                        
                    minutes = int(entry['time'] // 60)
                    seconds = entry['time'] % 60
                    time_str = f"{minutes}:{seconds:05.2f}"
                    name = entry['name'][:8].ljust(8)
                    text_content += f"{i:2d}{rank_emoji:2s} | {time_str:8s} | {name} | {entry['moves']:5d}\n"
        
        if not any(leaderboard.values()):
            text_content = "No entries in the leaderboard yet!"
        
        # Display the text
        plt.text(0.5, 0.5, text_content,
                horizontalalignment='center',
                verticalalignment='center',
                transform=plt.gca().transAxes,
                family='monospace')
        
        plt.title("Leaderboard (Press 'ESC' to close)")
        plt.draw()
        plt.show(block=False)
        
        # Connect escape key to close the leaderboard
        lb_fig.canvas.mpl_connect('key_press_event', 
                                 lambda event: plt.close(lb_fig) if event.key == 'escape' else None)
        
        logger.info("Leaderboard displayed")
    except Exception as e:
        logger.error(f"Error in display_leaderboard_window: {e}")

def display_terminal_leaderboard():
    """Display the leaderboard in the terminal with formatted output."""
    try:
        leaderboard = load_leaderboard()
        
        print("\n" + "="*50)
        print(" " * 20 + "LEADERBOARD")
        print("="*50)
        
        if not any(leaderboard.values()):
            print("\nNo entries in the leaderboard yet!")
            return
        
        for grid_key in sorted(leaderboard.keys()):
            if leaderboard[grid_key]:  # Only show sections with scores
                print(f"\n{grid_key} Grid:")
                print("-" * 50)
                print(f"{'Rank':<6} | {'Time':<10} | {'Name':<10} | {'Moves':<6}")
                print("-" * 50)
                
                for i, entry in enumerate(leaderboard[grid_key], 1):
                    # Add special emoji for top 3
                    rank_emoji = ""
                    if i == 1:
                        rank_emoji = "🥇"
                    elif i == 2:
                        rank_emoji = "🥈"
                    elif i == 3:
                        rank_emoji = "🥉"
                    
                    minutes = int(entry['time'] // 60)
                    seconds = entry['time'] % 60
                    time_str = f"{minutes}:{seconds:05.2f}"
                    name = entry['name'][:10].ljust(10)
                    
                    print(f"{i:2d}{rank_emoji:2s} | {time_str:10s} | {name} | {entry['moves']:6d}")
        
        print("\n" + "="*50)
        logger.info("Terminal leaderboard displayed")
    except Exception as e:
        logger.error(f"Error in display_terminal_leaderboard: {e}")

def on_key(event):
    """Handle keyboard events for controls."""
    try:
        if event.key == 'q':
            hide_hint()  # Hide hint when quitting
            plt.close('all')  # Close all figures
            return
        elif event.key == 'u':  # U for undo
            undo_move()
        elif event.key == 'y':  # Y for redo
            redo_move()
        elif event.key == 'r':  # R for restart
            restart_puzzle()
        elif event.key == 'h':  # H for hint
            show_hint()
        elif event.key == 'l':  # L for leaderboard
            display_leaderboard_window()
        elif event.key == 't':  # T for terminal leaderboard
            display_terminal_leaderboard()
        # Force redraw after any keyboard action
        plt.draw()
    except Exception as e:
        logger.error(f"Error in on_key: {e}")

def user_interaction():
    global fig, start_state, hint_usage_count
    try:
        # Reset hint usage count at start
        hint_usage_count = 0
        
        # Save initial state for restart
        start_state = {
            'pieces': pieces.copy(),
            'current_order': current_order.copy()
        }
        
        display_puzzle()
        # Connect keyboard events
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect("button_press_event", on_click)
        
        print("\nControls:")
        print("- Click on a piece to select it")
        print("- Click on another piece to swap them")
        print("- Press 'r' to restart the puzzle")
        print("- Press 'u' to undo last move")
        print("- Press 'y' to redo last move")
        print(f"- Press 'h' to show hint (original image) - {HINT_LIMIT} uses available")
        print("- Press 'l' to view leaderboard in window")
        print("- Press 't' to view leaderboard in terminal")
        print("- Press 'ESC' to hide hint/leaderboard")
        print("- Press 'q' to quit")
        
        plt.show()
    except Exception as e:
        logger.error(f"Error in user_interaction: {e}")
        raise

def display_leaderboard(current_grid_size=None):
    """Display the leaderboard with sections for different grid sizes."""
    leaderboard = load_leaderboard()
    
    if not any(leaderboard.values()):
        print("\nNo entries in the leaderboard yet!")
        return
    
    if current_grid_size:
        # Display only the current grid size section
        grid_key = f"{current_grid_size[0]}x{current_grid_size[1]}"
        if grid_key in leaderboard:
            display_grid_section(grid_key, leaderboard[grid_key])
    else:
        # Display all sections
        for grid_key in sorted(leaderboard.keys()):
            if leaderboard[grid_key]:  # Only show sections with scores
                display_grid_section(grid_key, leaderboard[grid_key])
                print()  # Add space between sections

def display_grid_section(grid_key, scores):
    """Display a single grid size section of the leaderboard."""
    print(f"\n=== LEADERBOARD ({grid_key}) ===")
    print("Rank | Time     | Name     | Moves")
    print("-" * 40)
    
    for i, entry in enumerate(scores, 1):
        # Add special emoji for top 3
        rank_emoji = ""
        if i == 1:
            rank_emoji = "🥇"
        elif i == 2:
            rank_emoji = "🥈"
        elif i == 3:
            rank_emoji = "🥉"
            
        minutes = int(entry['time'] // 60)
        seconds = entry['time'] % 60
        time_str = f"{minutes}:{seconds:05.2f}"
        name = entry['name'][:8].ljust(8)
        print(f"{i:2d}{rank_emoji:2s} | {time_str:8s} | {name} | {entry['moves']:5d}")

if __name__ == "__main__":
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description='Puzzle Solver Game')
        parser.add_argument('--leaderboard', '-lb', action='store_true',
                          help='Display the leaderboard and exit')
        args = parser.parse_args()

        # If --leaderboard flag is used, show leaderboard and exit
        if args.leaderboard:
            display_terminal_leaderboard()
            exit(0)

        # Normal game flow
        username = input("Enter your name: ")
        image_path = input("Enter image path: ").strip().strip('"')
        grid_size = tuple(map(int, input("Enter grid size (e.g., 3 3): ").split()))

        pieces, indices = split_image(image_path, grid_size)
        current_order = indices.copy()
        solved = False

        # Build and compile model
        model = build_cnn_model(input_shape=pieces.shape[1:], num_classes=len(pieces))

        start_time = time.time()
        user_interaction()

        if solved:
            end_time = time.time()
            time_taken = round(end_time - start_time, 2)
            logger.info(f"Puzzle solved in {time_taken} seconds and {moves} moves.")
            update_leaderboard(username, moves, time_taken, grid_size)
            print("\nCongratulations! Here's the updated leaderboard for your grid size:")
            display_leaderboard(grid_size)  # Show only current grid size
            print("\nFull Leaderboard across all grid sizes:")
            display_leaderboard()  # Show all grid sizes
        else:
            logger.info("Puzzle not solved.")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        raise