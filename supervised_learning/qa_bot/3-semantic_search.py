#!/usr/bin/env python3
"""
Semantic search using Sentence Transformer model
"""

import os
from sentence_transformers import SentenceTransformer, util


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents.

    Args:
        corpus_path: Path to the corpus of reference documents.
        sentence: The sentence to search for.

    Returns:
        The reference text of the most similar document.
    """
    # Load the Sentence Transformer model
    model = SentenceTransformer('all-mpnet-base-v2')

    # Embed the input sentence
    sentence_embedding = model.encode(sentence)

    # Initialize with a value lower than any possible similarity
    best_match = None
    best_similarity = -1

    # Iterate through the documents in the corpus
    for filename in os.listdir(corpus_path):
        if filename.endswith(".md"):
            filepath = os.path.join(corpus_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                reference_text = f.read()

                # Embed the reference text
                reference_embedding = model.encode(reference_text)

                # Calculate cosine similarity
                similarity = util.cos_sim(sentence_embedding,
                                          reference_embedding)
                # Extract the similarity score
                similarity = similarity.numpy()[0][0]

                # Update best match if a higher similarity is found
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = reference_text

    return best_match
