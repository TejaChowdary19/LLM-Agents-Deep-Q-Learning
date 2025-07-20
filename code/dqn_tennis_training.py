import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import ale_py
from ale_py import ALEInterface
import time
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class TennisEnv:
    """Wrapper for the ALE interface to use with the Tennis ROM"""
    def __init__(self, rom_path, display_screen=True):
        self.ale = ALEInterface()
        
        # Set ALE options
        self.ale.setBool('display_screen', display_screen)
        self.ale.setInt('random_seed', RANDOM_SEED)
        
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
        n_actions,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.000009,  # Faster decay for short training
        memory_size=10000,
        batch_size=32,
        update_target_freq=1000
    ):
        """Initialize the Deep Q-Network agent"""
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.update_target_freq = update_target_freq
        
        # Initialize replay memory buffer
        self.memory = deque(maxlen=memory_size)
        
        # Create main model (updated every step)
        self.model = self._build_model()
        
        # Create target model (updated every update_target_freq steps)
        self.target_model = self._build_model()
        self.update_target_network()
        
        # Training step counter
        self.training_steps = 0

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
        
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def update_target_network(self):
        """Update the target network with weights from the main network"""
        self.target_model.set_weights(self.model.get_weights())
        print("Target network updated")

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy"""
        if training and np.random.rand() < self.epsilon:
            # Random action (exploration)
            return np.random.randint(self.n_actions)
        else:
            # Predict Q values and select best action (exploitation)
            state_tensor = np.expand_dims(state, axis=0)
            q_values = self.model.predict(state_tensor, verbose=0)[0]
            return np.argmax(q_values)

    def train(self):
        """Train the model on a batch from replay memory"""
        # Ensure we have enough samples in memory
        if len(self.memory) < self.batch_size:
            return
            
        # Sample a batch of transitions from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract components from the batch
        states = np.array([transition[0] for transition in minibatch])
        actions = np.array([transition[1] for transition in minibatch])
        rewards = np.array([transition[2] for transition in minibatch])
        next_states = np.array([transition[3] for transition in minibatch])
        dones = np.array([transition[4] for transition in minibatch])
        
        # Compute target Q values
        target_q_values = self.target_model.predict(next_states, verbose=0)
        max_target_q = np.max(target_q_values, axis=1)
        
        # Apply Bellman equation: Q(s,a) = r + γ * max_a' Q(s',a')
        targets = rewards + (1 - dones) * self.gamma * max_target_q
        
        # Get current Q values from main model
        current_q = self.model.predict(states, verbose=0)
        
        # Update only the Q values for the actions taken
        for i, action in enumerate(actions):
            current_q[i][action] = targets[i]
        
        # Train the model with updated Q values
        self.model.fit(states, current_q, epochs=1, verbose=0)
        
        # Update epsilon for exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay
            
        # Increment training step counter
        self.training_steps += 1
        
        # Update target network if needed
        if self.training_steps % self.update_target_freq == 0:
            self.update_target_network()
            print(f"Step {self.training_steps}: Target network updated. Epsilon: {self.epsilon:.4f}")

    def save_model(self, filepath):
        """Save the model weights to a file"""
        # Ensure filepath ends with .weights.h5
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')
            if not filepath.endswith('.weights.h5'):
                filepath = filepath + '.weights.h5'
        self.model.save_weights(filepath)
        
    def load_model(self, filepath):
        """Load the model weights from a file"""
        # Ensure filepath ends with .weights.h5
        if not filepath.endswith('.weights.h5'):
            filepath = filepath.replace('.h5', '.weights.h5')
            if not filepath.endswith('.weights.h5'):
                filepath = filepath + '.weights.h5'
        self.model.load_weights(filepath)
        self.update_target_network()


def preprocess_state(state):
    """Preprocess the state (observation) for the DQN"""
    # Convert to grayscale
    gray = np.mean(state, axis=2).astype(np.uint8)
    
    # Resize to 84x84
    resized = tf.image.resize(np.expand_dims(gray, axis=-1), [84, 84])
    
    # Normalize to [0, 1]
    normalized = resized / 255.0
    
    return normalized.numpy()


def train_dqn(rom_path, total_episodes=50, max_steps=10000, display_screen=True):
    """Train a DQN agent on the Tennis environment"""
    # Create environment
    env = TennisEnv(rom_path, display_screen=display_screen)
    
    # Get state shape after preprocessing
    initial_state = env.reset()
    preprocessed_state = preprocess_state(initial_state)
    
    # Create state shape for stacked frames (4 frames stacked)
    stacked_state_shape = (84, 84, 4)
    
    print(f"Original state shape: {initial_state.shape}")
    print(f"Preprocessed state shape: {preprocessed_state.shape}")
    print(f"Stacked state shape: {stacked_state_shape}")
    print(f"Number of actions: {env.action_space}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_shape=stacked_state_shape,
        n_actions=env.action_space,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.000009,  # Faster decay for short training
        memory_size=10000,
        batch_size=32,
        update_target_freq=1000
    )
    
    # Training history
    training_history = {
        'episode_rewards': [],
        'epsilon_values': [],
        'avg_rewards': []
    }
    
    # Stack frames (initially with the same frame repeated)
    stacked_frames = np.repeat(preprocessed_state, 4, axis=2)
    
    # Train for specified number of episodes
    for episode in range(total_episodes):
        # Reset environment and get initial state
        state = env.reset()
        state = preprocess_state(state)
        
        # Reset frame stack with the initial state
        stacked_frames = np.repeat(state, 4, axis=2)
        
        episode_reward = 0
        
        # Track epsilon at the beginning of the episode
        training_history['epsilon_values'].append(agent.epsilon)
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(stacked_frames)
            next_state, reward, done, _ = env.step(action)
            
            # Preprocess next state
            next_state = preprocess_state(next_state)
            
            # Update stack of frames
            next_stacked_frames = np.copy(stacked_frames)
            next_stacked_frames = np.roll(next_stacked_frames, -1, axis=2)
            next_stacked_frames[:, :, -1] = next_state[:, :, 0]
            
            # Store transition in replay memory
            agent.store_transition(stacked_frames, action, reward, next_stacked_frames, done)
            
            # Update state and accumulated reward
            stacked_frames = next_stacked_frames
            episode_reward += reward
            
            # Train the agent
            agent.train()
            
            # Print step information
            if step % 100 == 0:
                print(f"\rEpisode {episode+1}/{total_episodes}, Step {step}/{max_steps}, Reward: {episode_reward}, Epsilon: {agent.epsilon:.4f}", end="")
            
            # End episode if done
            if done:
                break
        
        # Record episode reward
        training_history['episode_rewards'].append(episode_reward)
        
        # Calculate running average of episode rewards
        if len(training_history['episode_rewards']) > 10:
            avg_reward = np.mean(training_history['episode_rewards'][-10:])
        else:
            avg_reward = np.mean(training_history['episode_rewards'])
        
        training_history['avg_rewards'].append(avg_reward)
        
        # Print episode information
        print(f"\nEpisode {episode+1}/{total_episodes} - Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Save model every 10 episodes
        if (episode + 1) % 10 == 0:
            agent.save_model(f"dqn_tennis_episode_{episode+1}.weights.h5")
    
    # Save final model
    agent.save_model("dqn_tennis_final.weights.h5")
    
    return agent, training_history


def plot_training_history(history):
    """Plot training history metrics"""
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot episode rewards and running average
    episodes = range(1, len(history['episode_rewards']) + 1)
    ax1.plot(episodes, history['episode_rewards'], 'b-', alpha=0.3, label='Episode Reward')
    ax1.plot(episodes, history['avg_rewards'], 'r-', label='Running Avg')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot epsilon decay
    ax2.plot(episodes, history['epsilon_values'], 'g-')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate (Epsilon) Decay')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_tennis_training_history.png')
    plt.show()


def evaluate_agent(rom_path, model_path, n_episodes=5):
    """Evaluate a trained DQN agent"""
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
    agent.load_model(model_path)
    
    # Run evaluation episodes
    episode_rewards = []
    
    for episode in range(n_episodes):
        # Reset environment and get initial state
        state = env.reset()
        state = preprocess_state(state)
        
        # Reset frame stack with the initial state
        stacked_frames = np.repeat(state, 4, axis=2)
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done and step < 10000:
            # Select action without exploration
            action = agent.select_action(stacked_frames, training=False)
            
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
            
            # Add small delay for better visualization
            time.sleep(0.01)
        
        episode_rewards.append(episode_reward)
        print(f"Evaluation episode {episode+1}/{n_episodes}: Reward = {episode_reward}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average evaluation reward: {avg_reward:.2f}")
    return avg_reward


if __name__ == "__main__":
    # Get Tennis ROM path
    rom_path = os.path.join(ale_py.__path__[0], "roms", "tennis.bin")
    
    if not os.path.exists(rom_path):
        print(f"Tennis ROM not found at: {rom_path}")
        exit(1)
    
    print(f"Tennis ROM found at: {rom_path}")
    
    # Define training parameters
    TOTAL_EPISODES = 50  # Start with a small number for testing
    MAX_STEPS_PER_EPISODE = 5000
    DISPLAY_SCREEN = True  # Set to True to see the game while training
    
    # Train the agent
    print(f"Starting DQN training on Tennis...")
    start_time = time.time()
    
    agent, history = train_dqn(
        rom_path=rom_path,
        total_episodes=TOTAL_EPISODES,
        max_steps=MAX_STEPS_PER_EPISODE,
        display_screen=DISPLAY_SCREEN
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    evaluate_agent(rom_path, "dqn_tennis_final.weights.h5")