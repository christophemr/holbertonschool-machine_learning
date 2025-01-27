#!/usr/bin/env python3
"""defines the class Dataset that loads and preps a dataset
for machine translation
"""

import tensorflow.compat.v2 as tf  # type: ignore
import tensorflow_datasets as tfds
from transformers import AutoTokenizer


class Dataset:
    """
    Loads and preps a dataset for machine translation

    class constructor:
        def __init__(self)

    public instance attributes:
        data_train:
            contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset train split, loaded as_supervided
        data_valid:
            contains the ted_hrlr_translate/pt_to_en
                tf.data.Dataset validate split, loaded as_supervided
        tokenizer_pt:
            the Portuguese tokenizer created from the training set
        tokenizer_en:
            the English tokenizer created from the training set

    instance method:
        def tokenize_dataset(self, data):
            that creates sub-word tokenizers for our dataset
    """
    def __init__(self):
        """
        Class constructor.
        Initializes the train and validation datasets and tokenizers.
        """
        # Load the datasets
        self.data_train = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='train',
            as_supervised=True
        )
        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset.

        Args:
            data (tf.data.Dataset): Dataset whose examples are tuples (pt, en)
                - pt: tf.Tensor containing the Portuguese sentence
                - en: tf.Tensor containing the English sentence

        Returns:
            tokenizer_pt: Portuguese tokenizer
            tokenizer_en: English tokenizer
        """
        # Load pre-trained tokenizers
        tokenizer_pt = AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            vocab_size=2**13
        )
        tokenizer_en = AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en
