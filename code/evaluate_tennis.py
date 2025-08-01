import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import ale_py
from ale_py import ALEInterface
import time

# Get Tennis ROM path
rom_path = os.path.join(ale_py.__path__[0], "roms", "tennis.bin")

class TennisEnv:
    """Wrapper for the ALE interface to use with the Tennis ROM"""
    def __init__(self, rom_path, display_screen=True):
        self.ale = ALEInterface()
        
        # Set ALE options
        self.ale.setBool('display_screen', display_screen)
        
        # Load the ROM
        self.ale.loadROM(rom_path)
        
        # Get information about the game
        self.legal_actions = self.ale.getLegalActionSet()
        self.screen_dims = self.ale.getScreenDims()
        
        # Create a simple action space
        self.action_space = len(self.legal_actions)
        
    def reset(self):
        """Reset the game and return the initial state"""
        self.ale.reset_game()
        # Get the screen as RGB values
        screen = self.ale.getScreenRGB()
        return screen
        
    def step(self, action_idx):
        """Take a step with the given action index"""
        # Convert action index to ALE action
        action = self.legal_actions[action_idx]
        
        # Take action and get reward
        reward = self.ale.act(action)
        
        # Get the screen
        screen = self.ale.getScreenRGB()
        
        # Check if game is over
        done = self.ale.game_over()
        
        return screen, reward, done, {}
        
    def close(self):
        """Close the environment"""
        pass

class DQNAgent:
    def __init__(
        self,
        state_shape,
        n_actions
    ):
        """Initialize the Deep Q-Network agent"""
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Create model
        self.model = self._build_model()
        
    def _build_model(self):
        """Build a convolutional neural network model"""
        model = Sequential([
            # Input layer (84x84x4 grayscale frames)
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_shape),
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.n_actions, activation='linear')
        ])
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=0.00025))
        return model

    def load_model(self, filepath):
        """Load the model weights from a file"""
        # Ensure filepath ends with .weights.h5
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')
            if not filepath.endswith('.weights.h5'):
                filepath = filepath + '.weights.h5'
        
        self.model.load_weights(filepath)
        print(f"Model loaded from {filepath}")

    def select_action(self, state):
        """Select an action using the trained policy"""
        # Predict Q values
        state_tensor = np.expand_dims(state, axis=0)
        q_values = self.model.predict(state_tensor, verbose=0)[0]
        return np.argmax(q_values)

def preprocess_state(state):
    """Preprocess the state (observation) for the DQN"""
    # Convert to grayscale
    gray = np.mean(state, axis=2).astype(np.uint8)
    
    # Resize to 84x84
    resized = tf.image.resize(np.expand_dims(gray, axis=-1), [84, 84])
    
    # Normalize to [0, 1]
    normalized = resized / 255.0
    
    return normalized.numpy()

def evaluate():
    """Evaluate the trained agent"""
    # Create environment
    env = TennisEnv(rom_path, display_screen=True)
    
    # Get state shape after preprocessing
    initial_state = env.reset()
    preprocessed_state = preprocess_state(initial_state)
    
    # Create state shape for stacked frames (4 frames stacked)
    stacked_state_shape = (84, 84, 4)
    
    # Create DQN agent
    agent = DQNAgent(
        state_shape=stacked_state_shape,
        n_actions=env.action_space
    )
    
    # Load trained model
    agent.load_model("dqn_tennis_final.weights.h5")
    
    # Run evaluation episodes
    n_episodes = 5
    
    for episode in range(n_episodes):
        # Reset environment and get initial state
        state = env.reset()
        state = preprocess_state(state)
        
        # Reset frame stack with the initial state
        stacked_frames = np.repeat(state, 4, axis=2)
        
        episode_reward = 0
        done = False
        step = 0
        
        print(f"Episode {episode+1}/{n_episodes} - Starting evaluation...")
        
        while not done and step < 5000:
            # Select action using the trained policy
            action = agent.select_action(stacked_frames)
            
            # Take action in the environment
            next_state, reward, done, _ = env.step(action)
            
            # Preprocess next state
            next_state = preprocess_state(next_state)
            
            # Update stack of frames
            next_stacked_frames = np.copy(stacked_frames)
            next_stacked_frames = np.roll(next_stacked_frames, -1, axis=2)
            next_stacked_frames[:, :, -1] = next_state[:, :, 0]
            
            # Update state and reward
            stacked_frames = next_stacked_frames
            episode_reward += reward
            step += 1
            
            # Print step information periodically
            if step % 100 == 0:
                print(f"Step {step}, Reward so far: {episode_reward}")
            
            # Add small delay for better visualization
            time.sleep(0.01)
        
        print(f"Episode {episode+1}/{n_episodes} - Total Reward: {episode_reward}")
    
    env.close()
    print("Evaluation completed!")

if __name__ == "__main__":
    print(f"Evaluating trained DQN agent on Tennis...")
    evaluate()