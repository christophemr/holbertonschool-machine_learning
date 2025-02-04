#!/usr/bin/env python3
"""
implementation of a script that takes in input from the user
with the prompt Q: and prints A: as a response. If the user inputs
exit, quit, goodbye, or bye, case insensitive, print A: Goodbye and exit.
"""


def qa():
    """generates a loop to interact with the user
    exits the loop when the user types 'exit', 'quit', 'goodbye', or 'bye'.
    """
    while True:
        user_input = input("Q: ")

        # Convert to lowercase
        lower_input = user_input.lower()

        if lower_input in ("exit", "quit", "goodbye", "bye"):
            print("A: Goodbye")
            break
        else:
            print("A:")


# entry point
if __name__ == "__main__":
    qa()
