#!/usr/bin/python3
"""
Defines the training of an agent using Q-learning on the
FrozenLake environment.
"""

import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs Q-learning on the given FrozenLake environment.

    Args:
        env: The FrozenLakeEnv instance.
        Q (numpy.ndarray): The Q-table.
        episodes (int): Total number of episodes to train over.
        max_steps (int): Maximum number of steps per episode.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        epsilon (float): Initial threshold for epsilon-greedy.
        min_epsilon (float): Minimum value that epsilon should decay to.
        epsilon_decay (float): Decay rate for updating epsilon between episodes

    Returns:
        Q (numpy.ndarray): The updated Q-table.
        total_rewards (list): A list of total rewards per episode.
    """
    # list to store the total reward for each episode
    total_rewards = []

    # Loop over the specified number of episodes
    for episode in range(episodes):
        # Reset the environment at the beginning of each episode
        state = env.reset()[0]
        episode_reward = 0

        # For each episode, perform at most 'max_steps' steps
        for step in range(max_steps):
            # Choose an action using the epsilon-greedy strategy
            action = epsilon_greedy(Q, state, epsilon)

            # Take the chosen action in the environment
            next_state, reward, done, _, _ = env.step(action)

            # update the reward to -1 to penalize the failure
            if done and reward == 0:
                reward = -1

            # Update the Q-value for the current state and action
            Q[state, action] = Q[state, action] + alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state, action])

            # Accumulate the reward for this episode
            episode_reward += reward

            # Move to the next state
            state = next_state

            # If the episode is finished, exit the loop early
            if done:
                break

        # Decay the epsilon value, but do not let it fall below min_epsilon
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

        # Append the total reward obtained in this episode
        total_rewards.append(episode_reward)

    # Return the updated Q-table and the list of total rewards per episode
    return Q, total_rewards
