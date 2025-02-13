#!/usr/bin/env python3
"""
Defines function that creates a bag of words embedding matrix
"""


from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    Creates a bag of words embedding matrix

    parameters:
        sentences [list]:
            list of sentences to analyze

        vocab [list]:
            list of vocabulary words to use for analysis
            if None, all words within sentences should be used

    returns:
        embeddings,features:
            embeddings [numpy.ndarray of shape (s, f)]:
                contains the embeddings
                s: number of sentences in sentences
                f: number of features analyzed
            features [list]:
                list of features used for embeddings
    """
    vec = CountVectorizer(vocabulary=vocab)
    X = vec.fit_transform(sentences)
    embeddings = X.toarray()
    features = vec.get_feature_names_out()
    return embeddings, features
