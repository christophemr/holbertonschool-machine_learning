#!/usr/bin/env python3
"""Defines the class Dataset that loads and preps a dataset for machine translation."""

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
        def normalize_sentence(self, sentence):
            Normalizes spacing around punctuation in a sentence.
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
        ).take(10000).cache()

        self.data_valid = tfds.load(
            'ted_hrlr_translate/pt_to_en',
            split='validation',
            as_supervised=True
        )

        # Create tokenizers
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(self.data_train)

    def normalize_sentence(self, sentence):
        """
        Normalizes the spacing around punctuation in a sentence.
        Args:
            sentence (str): The sentence to normalize.
        Returns:
            str: The normalized sentence.
        """
        normalized = ""
        for i, char in enumerate(sentence):
            # Remove spaces before punctuation
            if char in ",.?!":
                if i > 0 and sentence[i - 1] == " ":
                    normalized = normalized[:-1]
            normalized += char
        return " ".join(normalized.split())

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
        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            "neuralmind/bert-base-portuguese-cased",
            vocab_size=2**13
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            "bert-base-uncased",
            vocab_size=2**13
        )

        # Normalize and compare a sample
        for pt, en in data.take(1):
            pt_sentence = pt.numpy().decode("utf-8").strip()
            en_sentence = en.numpy().decode("utf-8").strip()

            # Apply normalization
            pt_sentence = self.normalize_sentence(pt_sentence)
            en_sentence = self.normalize_sentence(en_sentence)

            # Tokenize and decode to verify consistency
            pt_tokenized = tokenizer_pt(pt_sentence, return_tensors="pt")["input_ids"]
            en_tokenized = tokenizer_en(en_sentence, return_tensors="pt")["input_ids"]

            pt_decoded = self.normalize_sentence(
                tokenizer_pt.decode(pt_tokenized[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            )
            en_decoded = self.normalize_sentence(
                tokenizer_en.decode(en_tokenized[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            )

        return tokenizer_pt, tokenizer_en
