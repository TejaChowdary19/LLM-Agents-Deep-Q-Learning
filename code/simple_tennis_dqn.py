# Save as simple_tennis_dqn.py
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import ale_py
import time
import matplotlib.pyplot as plt
from ale_py import ALEInterface

# Set random seeds for reproducibility
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# First, check if tennis.bin exists in ale_py roms directory
rom_dir = os.path.join(ale_py.__path__[0], "roms")
tennis_rom = os.path.join(rom_dir, "tennis.bin")

if os.path.exists(tennis_rom):
    print(f"Tennis ROM found at: {tennis_rom}")
else:
    print(f"Tennis ROM not found at: {tennis_rom}")
    print("Available ROMs:")
    for file in os.listdir(rom_dir):
        if file.endswith(".bin"):
            print(f"  - {file}")
    exit(1)

# Create a simple test using the Tennis ROM directly
ale = ALEInterface()

# Set display on
ale.setBool('display_screen', True)
ale.setInt('random_seed', 123)

# Load the Tennis ROM
print("Loading Tennis ROM...")
ale.loadROM(tennis_rom)

# Get information about the game
legal_actions = ale.getLegalActionSet()
print(f"Legal actions: {legal_actions}")
screen_dims = ale.getScreenDims()
print(f"Screen dimensions: {screen_dims}")

# Run a simple test with random actions
print("Running 100 random actions...")
ale.reset_game()
total_reward = 0

for i in range(100):
    # Choose a random action
    action = random.choice(legal_actions)
    
    # Take the action
    reward = ale.act(action)
    total_reward += reward
    
    # Check if game is over
    if ale.game_over():
        print(f"Game over at step {i+1}")
        ale.reset_game()
    
    # Print progress
    print(f"\rStep {i+1}/100, Action: {action}, Reward: {reward}, Total Reward: {total_reward}", end="")
    time.sleep(0.05)

print("\nTest completed!")