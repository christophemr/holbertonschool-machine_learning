#!/usr/bin/env python3
"""
Defines function that calculates the unigram BLEU score for a sentence
"""

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence

    Parameters:
        references (list of list of str): List of reference translations
        sentence (list of str): Model-proposed sentence

    Returns:
        float: Unigram BLEU score
    """
    # Step 1: Count matching words with clipping
    sentence_word_counts = {word: sentence.count(word) for word in sentence}
    clipped_counts = {}

    for word in sentence_word_counts:
        max_in_references = max(ref.count(word) for ref in references)
        clipped_counts[word] = min(
            sentence_word_counts[word], max_in_references)

    # Calculate precision (matches divided by total words in the sentence)
    total_matches = sum(clipped_counts.values())
    precision = total_matches / len(sentence)

    # Step 2: Compute brevity penalty
    sentence_length = len(sentence)
    reference_lengths = [len(ref) for ref in references]
    closest_ref_length = min(
        reference_lengths,
        key=lambda ref_len: (abs(ref_len - sentence_length), ref_len))

    if sentence_length > closest_ref_length:
        brevity_penalty = 1.0
    else:
        brevity_penalty = np.exp(1 - closest_ref_length / sentence_length)

    # Step 3: Calculate BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score
