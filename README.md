Deep Q-Learning Agent for Atari Tennis
This project implements a Deep Q-Learning (DQN) agent to play the Atari Tennis game using reinforcement learning techniques. The implementation follows the methodology established in DeepMind's seminal paper and applies it to the Tennis-v5 environment from OpenAI's Gymnasium.
Project Overview
This repository contains a complete implementation of a DQN agent for Atari Tennis, along with documentation, training results, and video demonstration as part of the "LLM Agents & Deep Q-Learning with Atari Games" assignment.
Repository Structure

dqn_tennis_training.py: Main implementation file with the DQN agent and training loop
test_tennis.py: Script to test the Tennis environment
evaluate_tennis.py: Script to evaluate the trained agent
record_video.py: Script to record a video demonstration of the trained agent
install_roms.py: Utility to install Atari ROMs
setup_atari.py: Environment setup utilities
Various model weights files (*.weights.h5): Saved model checkpoints
dqn_tennis_training_history.png: Plot of training performance
tennis_dqn_demo.mp4: Video demonstration of the trained agent
LICENSE: MIT license file
Documentation files (PDFs):

Deep Q-Learning Atari Agent Assignment Documentation.pdf: Technical documentation addressing all assignment requirements
Deep Q-Learning Agent for Atari Tennis-v5.pdf: Presentation slides explaining the implementation



Environment Setup
Prerequisites

Python 3.9+
TensorFlow 2.x
Gymnasium with Atari support
AutoROM for ROM installation
Additional packages: numpy, matplotlib, opencv-python

Installation

Create and activate a virtual environment:

bashpython3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install required packages:

bashpip3 install "gymnasium[atari,accept-rom-license]" "autorom[accept-rom-license]"
pip3 install tensorflow numpy matplotlib opencv-python

Install Atari ROMs:

bashpython install_roms.py
Usage
Training the Agent
bashpython dqn_tennis_training.py
The script will train the DQN agent on the Tennis environment using the specified hyperparameters. Training progress will be displayed in the console, and model checkpoints will be saved periodically.
Evaluating the Agent
bashpython evaluate_tennis.py
This will load the trained model and run evaluation episodes to demonstrate the agent's performance.
Recording a Video Demonstration
bashpython record_video.py
Creates a video recording of the trained agent playing Tennis.
Implementation Details
Agent Architecture

Neural Network: CNN-based architecture following DeepMind's DQN paper
Experience Replay: Buffer size of 100,000 transitions
Target Network: Updated every 10,000 steps
Frame Preprocessing: Grayscale conversion, resizing to 84x84, frame stacking (4 frames)

Hyperparameters

Learning Rate: 0.00025
Discount Factor (Gamma): 0.99
Exploration Rate (Epsilon): Start at 1.0, decay to 0.1
Epsilon Decay Rate: 0.995
Batch Size: 32

Results
The agent demonstrates learning capability in the Tennis environment, with performance improving over training episodes. The sparse reward structure of Tennis makes it a challenging environment, but the agent shows measurable improvement over random play.
Documentation
For detailed information about the implementation, analysis, and results, please refer to the included PDF documentation:

Assignment Documentation: Comprehensive answers to all assignment requirements
Presentation Slides: Visual explanation of the implementation and results

Video Demonstration
The file tennis_dqn_demo.mp4 contains a recorded demonstration of the trained agent playing Tennis, showing its learned behavior in action.
License
This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

PyTorch DQN tutorial: Adapted NN and training loop
Gymnasium + Atari: Used wrappers, preprocessing
AutoROM: Installed licensed ROMs

Author
Divya Teja Mannava - July 19, 2025
