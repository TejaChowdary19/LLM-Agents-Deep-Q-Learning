import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda
from tensorflow.keras.optimizers import Adam
from collections import deque
import gymnasium as gym
import time
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
RANDOM_SEED = 42
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class DQNAgent:
    def __init__(
        self,
        state_shape,
        n_actions,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.0000009,
        memory_size=100000,
        batch_size=32,
        update_target_freq=10000
    ):
        """Initialize the Deep Q-Network agent
        
        Args:
            state_shape: Shape of the state space
            n_actions: Number of possible actions
            learning_rate: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Rate at which epsilon decays
            memory_size: Size of the replay memory buffer
            batch_size: Size of batches for training
            update_target_freq: Frequency to update target network
        """
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
        """Build a convolutional neural network model
        
        Returns:
            A compiled Keras model
        """
        # Create model based on the DeepMind DQN architecture
        model = Sequential([
            # Transpose input from (batch, stack, height, width) to (batch, height, width, stack)
            Lambda(lambda x: tf.transpose(x, [0, 2, 3, 1]), input_shape=self.state_shape),
            
            # Convolutional layers
            Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
            Conv2D(64, (4, 4), strides=(2, 2), activation='relu'),
            Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
            
            # Flatten and dense layers
            Flatten(),
            Dense(512, activation='relu'),
            Dense(self.n_actions, activation='linear')
        ])
        
        model.compile(loss='huber_loss', optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0))
        return model

    def update_target_network(self):
        """Update the target network with weights from the main network"""
        self.target_model.set_weights(self.model.get_weights())

    def store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the replay memory buffer
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        self.memory.append((state, action, reward, next_state, done))

    def select_action(self, state, training=True):
        """Select an action using epsilon-greedy policy
        
        Args:
            state: Current state
            training: Whether in training mode (affects exploration)
            
        Returns:
            The selected action
        """
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
            print(f"Target network updated. Epsilon: {self.epsilon:.4f}")

    def save_model(self, filepath):
        """Save the model weights to a file
        
        Args:
            filepath: Path to save the model
        """
        self.model.save_weights(filepath)
        
    def load_model(self, filepath):
        """Load the model weights from a file
        
        Args:
            filepath: Path to load the model from
        """
        self.model.load_weights(filepath)
        self.update_target_network()


def evaluate_agent(agent, env, n_episodes=10):
    """Evaluate the agent's performance without exploration
    
    Args:
        agent: The DQN agent
        env: The gym environment
        n_episodes: Number of episodes to evaluate
        
    Returns:
        Average reward across episodes
    """
    episode_rewards = []
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action without exploration
            action = agent.select_action(state, training=False)
            
            # Take action in the environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
        episode_rewards.append(episode_reward)
        print(f"Evaluation episode {episode+1}/{n_episodes}: Reward = {episode_reward}")
    
    avg_reward = np.mean(episode_rewards)
    print(f"Average evaluation reward: {avg_reward:.2f}")
    return avg_reward


def train_dqn(env_name, total_episodes=5000, max_steps=10000):
    """Train a DQN agent on an Atari environment
    
    Args:
        env_name: Name of the Atari environment
        total_episodes: Maximum number of episodes to train
        max_steps: Maximum steps per episode
        
    Returns:
        Trained DQN agent and training history
    """
    # Create Atari environment
    env = gym.make(env_name)
    
    # Get state and action space information
    state, _ = env.reset()
    state_shape = state.shape
    n_actions = env.action_space.n
    
    print(f"State shape: {state_shape}")
    print(f"Number of actions: {n_actions}")
    
    # Create DQN agent
    agent = DQNAgent(
        state_shape=state_shape,
        n_actions=n_actions,
        learning_rate=0.00025,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.1,
        epsilon_decay=0.0000009,  # Slower decay for more exploration
        memory_size=100000,
        batch_size=32,
        update_target_freq=10000
    )
    
    # Training history
    training_history = {
        'episode_rewards': [],
        'epsilon_values': [],
        'avg_rewards': []
    }
    
    # Testing parameters
    test_episodes = 10
    test_frequency = 100  # Test every 100 episodes
    
    # Train for specified number of episodes
    for episode in range(total_episodes):
        # Reset environment and get initial state
        state, _ = env.reset()
        episode_reward = 0
        
        # Track epsilon at the beginning of the episode
        training_history['epsilon_values'].append(agent.epsilon)
        
        for step in range(max_steps):
            # Select and perform action
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition in replay memory
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update state and accumulated reward
            state = next_state
            episode_reward += reward
            
            # Train the agent
            agent.train()
            
            # End episode if done
            if done:
                break
        
        # Record episode reward
        training_history['episode_rewards'].append(episode_reward)
        
        # Calculate running average of episode rewards (last 100 episodes)
        if len(training_history['episode_rewards']) > 100:
            avg_reward = np.mean(training_history['episode_rewards'][-100:])
        else:
            avg_reward = np.mean(training_history['episode_rewards'])
        
        training_history['avg_rewards'].append(avg_reward)
        
        # Print episode information
        print(f"Episode {episode+1}/{total_episodes} - Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
        
        # Evaluate agent performance periodically
        if (episode + 1) % test_frequency == 0:
            # Create a test environment with rendering
            test_env = gym.make(env_name, render_mode="human")
            
            print(f"\nEvaluating agent after {episode+1} episodes...")
            evaluate_agent(agent, test_env, n_episodes=test_episodes)
            print("\nResuming training...\n")
            
            # Save model weights
            agent.save_model(f"dqn_{env_name.split('-')[0].lower()}_{episode+1}.h5")
    
    # Final evaluation
    test_env = gym.make(env_name, render_mode="human")
    
    print("\nFinal evaluation:")
    evaluate_agent(agent, test_env, n_episodes=test_episodes)
    
    # Save final model
    agent.save_model(f"dqn_{env_name.split('-')[0].lower()}_final.h5")
    
    return agent, training_history


def plot_training_history(history):
    """Plot training history metrics
    
    Args:
        history: Dictionary containing training metrics
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot episode rewards and running average
    episodes = range(1, len(history['episode_rewards']) + 1)
    ax1.plot(episodes, history['episode_rewards'], 'b-', alpha=0.3, label='Episode Reward')
    ax1.plot(episodes, history['avg_rewards'], 'r-', label='Running Avg (100 episodes)')
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
    plt.savefig('dqn_training_history.png')
    plt.show()


if __name__ == "__main__":
    # Select Atari game environment
    ENV_NAME = "ALE/Tennis-v5"  # Using Tennis as our target game
    
    # Training parameters
    TOTAL_EPISODES = 1000  # Reduced for initial testing; increase for better results
    MAX_STEPS_PER_EPISODE = 10000
    
    # Start training
    print(f"Starting DQN training on {ENV_NAME}...")
    start_time = time.time()
    
    agent, history = train_dqn(
        env_name=ENV_NAME,
        total_episodes=TOTAL_EPISODES,
        max_steps=MAX_STEPS_PER_EPISODE
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Plot training history
    plot_training_history(history)
    
    # Create a visualization environment
    env = gym.make(ENV_NAME, render_mode="human")
    
    # Run visualization
    print("\nRunning visualization of trained agent...")
    state, _ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        # Select action using the trained policy
        action = agent.select_action(state, training=False)
        
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update state and reward
        state = next_state
        total_reward += reward
        
        # Add small delay for better visualization
        time.sleep(0.01)
    
    print(f"Visualization complete. Total reward: {total_reward}")
    env.close()