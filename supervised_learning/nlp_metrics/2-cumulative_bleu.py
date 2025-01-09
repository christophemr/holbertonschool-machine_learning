#!/usr/bin/env python3
"""
Defines function that calculates cumulative n-gram BLEU score for a sentence
"""

import numpy as np


def generate_ngrams(sequence, n):
    """Generate n-grams from a sequence."""
    return [" ".join(sequence[i:i + n]) for i in range(len(sequence) - n + 1)]


def precision(references, sentence, n):
    """
    Calculates precision for n-gram BLEU score.
    """
    # Generate n-grams for sentence
    sentence_ngrams = generate_ngrams(sentence, n)
    sentence_counts = (
        {ngram: sentence_ngrams.count(ngram) for ngram in sentence_ngrams})

    # Generate n-grams for references and compute max counts
    max_counts = {}
    for ref in references:
        ref_ngrams = generate_ngrams(ref, n)
        ref_counts = {ngram: ref_ngrams.count(ngram) for ngram in ref_ngrams}
        for ngram in ref_counts:
            max_counts[ngram] = max(
                max_counts.get(ngram, 0), ref_counts[ngram])

    # Compute clipped counts
    clipped_counts = (
        {ngram: min(count, max_counts.get(ngram, 0)) for ngram,
         count in sentence_counts.items()})

    # Calculate precision
    total_ngrams = len(sentence_ngrams)
    total_clipped = sum(clipped_counts.values())

    return total_clipped / total_ngrams if total_ngrams > 0 else 0


def cumulative_bleu(references, sentence, n):
    """
    Calculates the cumulative n-gram BLEU score for a sentence.
    """
    # Compute precisions for all n-grams
    precisions = []
    for i in range(1, n + 1):
        precisions.append(precision(references, sentence, i))

    # Handle edge case where any precision is zero
    if min(precisions) == 0:
        return 0

    # Compute geometric mean of precisions
    geometric_mean = np.exp(np.mean(np.log(precisions)))

    # Compute brevity penalty
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
    bleu_score = brevity_penalty * geometric_mean

    return bleu_score
