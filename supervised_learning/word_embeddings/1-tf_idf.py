#!/usr/bin/env python3
"""
Defines function that creates a TF-IDF embedding
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    Creates a TF-IDF embedding.

    Parameters:
        sentences [list]: List of sentences to analyze
        vocab [list]: List of vocabulary words to use for analysis
                      If None, all words within sentences should be used

    Returns:
        embeddings [numpy.ndarray of shape (s, f)]: Contains the embeddings
            s: Number of sentences in sentences
            f: Number of features analyzed
        features [list]: List of features used for embeddings
    """
    vec = TfidfVectorizer(vocabulary=vocab)
    X = vec.fit_transform(sentences)
    embeddings = X.toarray()
    features = vec.get_feature_names_out()
    return embeddings, features
