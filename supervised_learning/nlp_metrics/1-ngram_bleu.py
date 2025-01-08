#!/usr/bin/env python3
"""
Defines function that calculates the n-gram BLEU score for a sentence
"""

import numpy as np


def generate_ngrams(sequence, n):
    """Generate n-grams from a sequence."""
    return [" ".join(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def ngram_bleu(references, sentence, n):
    """
    Calculates the n-gram BLEU score for a sentence

    Parameters:
        references (list of list of str): Reference translations
        sentence (list of str): Model-proposed sentence
        n (int): Size of the n-gram to use for evaluation

    Returns:
        float: n-gram BLEU score
    """
    # Generate n-grams for sentence and references
    sentence_ngrams = generate_ngrams(sentence, n)
    sentence_counts = {}
    for ngram in sentence_ngrams:
        sentence_counts[ngram] = sentence_counts.get(ngram, 0) + 1

    # Maximum n-gram counts in references
    max_reference_counts = {}
    for reference in references:
        reference_ngrams = generate_ngrams(reference, n)
        reference_counts = {}
        for ngram in reference_ngrams:
            reference_counts[ngram] = reference_counts.get(ngram, 0) + 1
        for ngram, count in reference_counts.items():
            max_reference_counts[ngram] = max(
                max_reference_counts.get(ngram, 0), count)

    # Calculate clipped counts
    total_clipped = sum(
        min(count, max_reference_counts.get(ngram, 0))
        for ngram, count in sentence_counts.items()
    )
    total_ngrams = len(sentence_ngrams)

    # Calculate precision
    precision = total_clipped / total_ngrams if total_ngrams > 0 else 0

    # Calculate brevity penalty
    sentence_length = len(sentence)
    reference_lengths = [len(ref) for ref in references]
    closest_ref_length = min(
        reference_lengths,
        key=lambda ref_len: (abs(ref_len - sentence_length), ref_len))

    if sentence_length > closest_ref_length:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_length / sentence_length)

    # Calculate BLEU score
    bleu_score = brevity_penalty * precision

    return bleu_score
