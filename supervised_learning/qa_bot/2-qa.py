#!/usr/bin/env python3
"""
Function that answers questions from a reference text
"""
question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Answers questions from a reference text.

    Args:
        reference: The reference text.
    """
    while True:
        question = input("Q: ")
        # remove whitespaces and convert to lowercase
        question = question.strip().lower()

        # check if user wants to quit
        if question in ("exit", "quit", "goodbye", "bye"):
            print("A: Goodbye")
            break

        # retrieve answer using qa function
        answer = question_answer(question, reference)

        # if no match
        if answer is None:
            print("A: Sorry, I do not understand your question.")
        else:
            print(f"A: {answer}")
