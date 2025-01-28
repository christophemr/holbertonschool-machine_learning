#!/usr/bin/env python3
"""Defines the class Dataset that loads and preps a dataset
for machine translation."""

import tensorflow_datasets as tfds
import transformers


class Dataset:
    """
    Loads and preps a dataset for machine translation.

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
            Creates sub-word tokenizers for our dataset.
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
        # Extract Portuguese and English sentences
        pt_sentences, en_sentences = list(zip(*data.as_numpy_iterator()))
        pt_sentences = [sentence.decode('utf-8') for sentence in pt_sentences]
        en_sentences = [sentence.decode('utf-8') for sentence in en_sentences]

        # Load pre-trained tokenizers from Transformers
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased', use_fast=True)
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased', use_fast=True)

        # Train the tokenizers on the extracted sentences
        tokenizer_pt.train_new_from_iterator(pt_sentences, vocab_size=2**13,
                                             clean_up_tokenization_spaces=True)
        tokenizer_en.train_new_from_iterator(en_sentences, vocab_size=2**13,
                                             clean_up_tokenization_spaces=True)

        return tokenizer_pt, tokenizer_en
