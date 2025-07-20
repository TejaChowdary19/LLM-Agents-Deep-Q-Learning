# Save as install_test_atari.py
import os
import sys
import subprocess
import time

print("=== Installing required packages ===")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "gymnasium[atari]", "ale-py", "autorom[accept-rom-license]"])

print("\n=== Checking if ROM installation is needed ===")
try:
    import ale_py
    rom_path = os.path.join(ale_py.__path__[0], "roms")
    if os.path.exists(rom_path) and len(os.listdir(rom_path)) > 0:
        print(f"ROMs already exist at: {rom_path}")
    else:
        print("ROMs not found, attempting to install...")
        try:
            from autorom.accept_rom_license import accept_license_and_download
            rom_directory = "./roms"
            os.makedirs(rom_directory, exist_ok=True)
            accept_license_and_download(rom_directory)
            print(f"ROMs downloaded to: {rom_directory}")
            
            # Try to copy ROMs to ale_py directory
            print(f"Copying ROMs to {rom_path}...")
            os.makedirs(rom_path, exist_ok=True)
            for rom_file in os.listdir(rom_directory):
                if rom_file.endswith(".bin"):
                    src = os.path.join(rom_directory, rom_file)
                    dst = os.path.join(rom_path, rom_file)
                    import shutil
                    shutil.copy2(src, dst)
                    print(f"Copied {rom_file}")
        except Exception as e:
            print(f"Error downloading ROMs: {e}")
except Exception as e:
    print(f"Error checking ROMs: {e}")

print("\n=== Testing Atari Environment ===")
try:
    import gymnasium as gym
    env = gym.make("ALE/Breakout-v5", render_mode="human")
    print(f"Successfully created environment with action space: {env.action_space}")
    
    observation, info = env.reset()
    print(f"Observation shape: {observation.shape}")
    
    for i in range(20):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}, Action: {action}, Reward: {reward}")
        time.sleep(0.1)
        
        if terminated or truncated:
            observation, info = env.reset()
    
    env.close()
    print("Atari test successful!")
except Exception as e:
    print(f"Error testing Atari environment: {e}")

print("\nSetup and test complete!")