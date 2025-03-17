#!/usr/bin/env python3
"""
creates a Pandas DataFrame from a dictionary and saves it into variable df
"""

import pandas as pd


# Define the dictionary
data = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ["one", "two", "three", "four"]
}

# Create the DataFrame
df = pd.DataFrame(data, index=["A", "B", "C", "D"])
