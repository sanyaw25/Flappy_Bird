
# FLAPPY BIRD

<p align="center">
  <a href="#about">About</a> &nbsp;&bull;&nbsp;
  <a href="#solution-architecture">The Solution Architecture</a> &nbsp;&bull;&nbsp;
  <a href="#research">Research</a> &nbsp;&bull;&nbsp;
  <a href="#getting-started">Getting Started</a> &nbsp;&bull;&nbsp;
  <a href="#tech-stack">Tech Stack</a>
</p>

<br>

<div align="center">
  <img src="https://upload.wikimedia.org/wikipedia/en/0/0a/Flappy_Bird_icon.png" alt="Flappy Bird" width="400"/>
</div>

## 🏆 About

**Training Flappy Bird using Reinforcement Learning**
This project applies reinforcement learning to the classic Flappy Bird game using Stable Baselines3 and OpenAI Gym. It includes training a Deep Q-Network (DQN) to master the game and evaluate its performance over time. 

## 📂 The Solution Architecture

### Project Overview

- **Environment:** Flappy Bird Gym Environment
- **Algorithm:** Deep Q-Learning (DQN)
- **Neural Network:** Custom-built with Keras
- **Visualization:** Training progress and performance metrics plotted using Matplotlib

### Key Components

- **Neural Network Model:** Built with Keras, consisting of multiple dense layers.
- **Callbacks:** Custom callbacks for saving model checkpoints and plotting training metrics.
- **Training Script:** Executes the training process and monitors performance.

## 🔬 Research

For insights into reinforcement learning and the DQN algorithm, refer to these resources:

- [Deep Q-Learning](https://arxiv.org/abs/1312.5602)
- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [OpenAI Gym Documentation](https://gym.openai.com/docs/)

## 🚀 Getting Started

To get started with this project, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/flappy-bird-reinforcement-learning.git
cd flappy-bird-reinforcement-learning
```

### 2. Install Dependencies

Install the required libraries and packages:

```bash
pip install pygame keras-rl2 stable-baselines3[extra] gymnasium
```

### 3. Set Up and Train the Model

Run the following script to set up the environment, build the neural network, and train the model:

```python
import time
import numpy as np
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib.pyplot as plt
import flappy_bird_gym

# Define the environment
env = flappy_bird_gym.make("FlappyBird-v0", new_step_api=True)

# Define the neural network model
def build_model(input_shape, actions):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(1, input_shape)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Flatten())
    model.add(Dense(actions, activation='linear'))
    model.summary()
    return model

# Build the model
obs = env.observation_space.shape[0]
actions = env.action_space.n
model = build_model(obs, actions)

# Define the Plotting Callback
class PlottingCallback(BaseCallback):
    def __init__(self, save_freq, verbose=1):
        super(PlottingCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.rewards = []
        self.steps = []
        self.cumulative_rewards = []

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            # Collect metrics
            env = self.model.get_env()
            mean_reward, _ = evaluate_policy(self.model, env, n_eval_episodes=10, return_episode_rewards=False)
            self.rewards.append(mean_reward)
            self.steps.append(self.num_timesteps)

            # Track cumulative reward
            self.cumulative_rewards.append(mean_reward * 10)  # Assuming 10 episodes

            # Plot metrics
            fig, axs = plt.subplots(2, 1, figsize=(14, 8))

            # Mean Reward
            axs[0].plot(self.steps, self.rewards, label='Mean Reward', color='blue')
            axs[0].set_xlabel('Timesteps')
            axs[0].set_ylabel('Mean Reward')
            axs[0].set_title('Mean Reward Over Time')
            axs[0].legend()
            axs[0].grid()

            # Cumulative Reward
            axs[1].plot(self.steps, self.cumulative_rewards, label='Cumulative Reward', color='orange')
            axs[1].set_xlabel('Timesteps')
            axs[1].set_ylabel('Cumulative Reward')
            axs[1].set_title('Cumulative Reward Over Time')
            axs[1].legend()
            axs[1].grid()

            plt.tight_layout()
            plt.show()

        return True

# Create the environment
env = DummyVecEnv([lambda: gym.make('FlappyBird-v0')])

# Build the DQN model
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=0.00025,
    buffer_size=100000,
    batch_size=32,
    tau=1.0,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.01,
    max_grad_norm=10,
    verbose=1,
)

# Define the checkpoint callback
checkpoint_callback = CheckpointCallback(
    save_freq=40000,
    save_path='/content/drive/MyDrive/Flap/',
    name_prefix='checkpoint_Sanya' #name of the checkpoint
)

# Define the callback for plotting
plotting_callback = PlottingCallback(save_freq=10000)

# Train the model with both callbacks
model.learn(total_timesteps=2000000, log_interval=10, callback=[checkpoint_callback, plotting_callback])

# Clean up the environment after training
env.close()
```

## 🛠 Tech Stack

- **Python Libraries:** TensorFlow, Keras, Stable Baselines3, OpenAI Gym, Matplotlib, Pygame
- **Reinforcement Learning:** Deep Q-Learning (DQN)
- **Environment:** Flappy Bird Gym Environment

## 📈 Results

The model's performance is tracked and visualized with graphs showing the mean reward and cumulative reward over time using matplotlib.

feel free to open an issue or contribute to the project!


## 📧 Contact

For any questions or feedback, feel free to reach out at [sanyaw12722@gmail.com](mailto:your-email@example.com).


## 🔧 Troubleshooting

**Common Issues:**
- Issue: "Error while installing dependencies"
  - Solution: Ensure you have the correct version of Python and all required libraries that are compatible.


## 🚧 Roadmap

- **Future Improvements:** Will add visuals and sounds for the game.

## 🤝 Acknowledgments

- Thanks to the creators of [Flappy Bird](https://en.wikipedia.org/wiki/Flappy_Bird) and the developers of [Stable Baselines3](https://stable-baselines3.readthedocs.io/) and https://github.com/samboylan11/flappy-bird-gym for the inspiration and environment.
```
