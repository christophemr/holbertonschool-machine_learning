#!/usr/bin/env python3
"""
Defines function that finds a snippet of text within a reference
document to answer a question
"""

import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer


def question_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer
    a question.

    Args:
        question: A string containing the question to answer.
        reference: A string containing the reference document.

    Returns:
        A string containing the answer, or None if no answer is found.
    """
    # load BERT QA model
    model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")

    # load tokenizer
    tokenizer = BertTokenizer.from_pretrained(
        'bert-large-uncased-whole-word-masking-finetuned-squad')

    # tokenize the documents
    inputs = tokenizer(question, reference, return_tensors="tf")

    input_tensors = [
        inputs["input_ids"],
        inputs["attention_mask"],
        inputs["token_type_ids"]
    ]
    # run model
    output = model(input_tensors)

    # get start and end logits
    start_logits = output[0]
    end_logits = output[1]

    # Get the input sequence length
    sequence_length = inputs["input_ids"].shape[1]

    # Get the start and end indexes with the highest scores
    start_index = tf.math.argmax(start_logits[0, 1:sequence_length - 1]) + 1
    end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

    # Get the answer tokens using the best indices
    answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

    # decode the answer
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True)

    # return None if no valid answer
    if not answer.strip():
        return None

    return answer
