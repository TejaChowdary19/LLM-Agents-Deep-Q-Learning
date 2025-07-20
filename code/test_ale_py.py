# Save as test_ale_py.py
import ale_py
import os

# Print ALE-py version
print(f"ALE-py version: {ale_py.__version__}")

# List ROM directory
rom_dir = os.path.join(ale_py.__path__[0], "roms")
print(f"ROM directory: {rom_dir}")

if os.path.exists(rom_dir):
    print("ROM files found:")
    for rom in os.listdir(rom_dir):
        print(f"  - {rom}")
else:
    print("ROM directory not found!")

# Try to create an ALE environment directly
try:
    from ale_py import ALEInterface
    ale = ALEInterface()
    
    # Try to load a ROM
    rom_files = [f for f in os.listdir(rom_dir) if f.endswith('.bin')]
    if rom_files:
        test_rom = os.path.join(rom_dir, rom_files[0])
        print(f"\nTrying to load ROM: {test_rom}")
        ale.loadROM(test_rom)
        print("ROM loaded successfully!")
        
        # Get game info
        print(f"Game: {ale.getGameName()}")
        print(f"Frame dimensions: {ale.getScreenDims()}")
        print(f"Lives: {ale.lives()}")
        print(f"Available actions: {ale.getLegalActionSet()}")
        
        # Take some random actions
        import random
        actions = ale.getLegalActionSet()
        for i in range(10):
            action = random.choice(actions)
            reward = ale.act(action)
            print(f"Step {i+1}, Action: {action}, Reward: {reward}")
    else:
        print("No ROM files found!")
except Exception as e:
    print(f"Error with ALEInterface: {e}")

print("\nTest complete!")