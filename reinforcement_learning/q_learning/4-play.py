#!/usr/bin/env python3
"""
Plays one episode using the trained agent's Q-table on the
FrozenLake environment.
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Simulates an episode using a trained Q-table, exploiting the learned
    policy.

    Parameters:
    ----------
    env : gym.Env
        The FrozenLake environment instance.
    Q : numpy.ndarray
        The trained Q-table.
    max_steps : int, optional
        Maximum number of steps in the episode.

    Returns:
    -------
    total_rewards : float
        Total rewards obtained in the episode.
    rendered_outputs : list of str
        List of environment renderings for each step.
    """
    # initialize the environment and get the initial state
    state = env.reset()[0]

    total_rewards = 0
    # get the initial state
    rendered_outputs = [env.render()]

    for step in range(max_steps):
        # Choose the action based on highest Q-value
        action = np.argmax(Q[state])

        # Take the action in the environment
        next_state, reward, done, _, _ = env.step(action)

        # Update total rewards
        total_rewards += reward

        # Capture the current state of the environment
        rendered_outputs.append(env.render())

        # update the state to the next state
        state = next_state

        # If the episode is finished, exit the loop
        if done:
            break

    return total_rewards, rendered_outputs
