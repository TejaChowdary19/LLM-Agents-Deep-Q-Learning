# Save this as setup_atari.py
import os
import sys
import subprocess

def check_package(package_name):
    try:
        __import__(package_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is NOT installed")
        return False

print("\n=== Checking installed packages ===")
gymnasium_installed = check_package("gymnasium")
ale_py_installed = check_package("ale_py")
autorom_installed = check_package("autorom")

if not gymnasium_installed or not ale_py_installed:
    print("\nInstalling required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gymnasium[atari]"])
    
if not autorom_installed:
    print("\nInstalling AutoROM...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "autorom", "autorom.accept-rom-license"])

print("\n=== Trying to download ROMs ===")
try:
    subprocess.check_call([sys.executable, "-m", "autorom", "--accept-license", "--install-dir", "./roms"])
    print("✅ ROMs downloaded successfully")
except Exception as e:
    print(f"❌ ROM download failed: {e}")
    
print("\n=== Testing Atari environments ===")
try:
    import gymnasium as gym
    # Try to create a simple Atari environment
    try:
        env = gym.make("ALE/Breakout-v5")
        print(f"✅ Successfully created ALE/Breakout-v5 environment with action space: {env.action_space}")
        env.close()
    except Exception as e:
        print(f"❌ Error creating Breakout environment: {e}")
        
    # Try to list all available environments
    from gymnasium.envs.registration import registry
    print("\nAvailable ALE environments:")
    ale_envs = [env_id for env_id in registry.keys() if 'ALE' in env_id]
    for i, env_id in enumerate(ale_envs[:10]):  # Show only first 10
        print(f"  {i+1}. {env_id}")
    if len(ale_envs) > 10:
        print(f"  ... and {len(ale_envs) - 10} more")
        
except Exception as e:
    print(f"❌ Error testing environments: {e}")

print("\nSetup complete!")