#!/usr/bin/env python3

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loads the pre-made FrozenLake environment from gymnasium.

    Args:
        desc (list of lists, optional): A custom map description.
            If provided, this list of lists will define the environment's
            layout.
        map_name (str, optional): The name of a pre-made map to load.
        is_slippery (bool): Determines whether the ice is slippery.
            Defaults to False.
    Returns:
        The FrozenLake environment.

    """
    environment = gym.make("FrozenLake-v1",
                           desc=desc, map_name=map_name,
                           is_slippery=is_slippery)

    return environment
