import gymnasium as gym
import ale_py
import os
import time

# First, verify that the tennis ROM exists
rom_dir = os.path.join(ale_py.__path__[0], "roms")
tennis_rom = os.path.join(rom_dir, "tennis.bin")

if os.path.exists(tennis_rom):
    print(f"Tennis ROM found at: {tennis_rom}")
else:
    print("Tennis ROM not found!")
    exit(1)

# Try different approaches to run the Tennis game

# Approach 1: Try using Gymnasium with ALE prefix
try:
    print("\nApproach 1: Using ALE prefix")
    env = gym.make("ALE/Tennis-v5", render_mode="human")
    print("Success!")
    
    observation, info = env.reset()
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    
    for i in range(50):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"\rStep {i+1}/50, Action: {action}, Reward: {reward}", end="")
        time.sleep(0.05)
        
        if terminated or truncated:
            observation, info = env.reset()
            print("\nEpisode ended, resetting...")
    
    env.close()
    print("\nApproach 1 successful!")
except Exception as e:
    print(f"\nApproach 1 failed: {e}")

# Approach 2: Try using the atari_py interface directly
try:
    print("\nApproach 2: Using ale_py directly")
    from ale_py import ALEInterface
    
    ale = ALEInterface()
    ale.loadROM(tennis_rom)
    
    # Get available actions
    actions = ale.getLegalActionSet()
    print(f"Legal actions: {actions}")
    
    # Take some random actions
    import random
    for i in range(20):
        action = random.choice(actions)
        reward = ale.act(action)
        print(f"Step {i+1}, Action: {action}, Reward: {reward}")
        
        # Display the screen (optional)
        screen = ale.getScreenRGB()
        # To display the screen, you would need a way to render it (e.g., pygame)
        
    print("Approach 2 successful!")
except Exception as e:
    print(f"Approach 2 failed: {e}")

# Approach 3: Try using a simpler Atari environment
try:
    print("\nApproach 3: Using a simpler Atari environment (Pong)")
    env = gym.make("Pong-v0", render_mode="human")
    
    observation, info = env.reset()
    print(f"Observation shape: {observation.shape}")
    print(f"Action space: {env.action_space}")
    
    for i in range(20):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"\rStep {i+1}/20, Action: {action}, Reward: {reward}", end="")
        time.sleep(0.05)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
    print("\nApproach 3 successful!")
except Exception as e:
    print(f"\nApproach 3 failed: {e}")

print("\nTest complete!")