#!/usr/bin/env python3
"""
defines a function that answers questions from multiple reference texts
"""

import os
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer
from sentence_transformers import SentenceTransformer, util


def retrieve_answer(question, reference):
    """
    Finds a snippet of text within a reference document to answer
    a question.

    Args:
        question: A string containing the question to answer.
        reference: A string containing the reference document.

    Returns:
        A string containing the answer, or None if no answer is found.
    """
    try:
        model = hub.load("https://tfhub.dev/see--/bert-uncased-tf2-qa/1")
        tokenizer = BertTokenizer.from_pretrained(
            'bert-large-uncased-whole-word-masking-finetuned-squad')

        inputs = tokenizer(question, reference, return_tensors="tf")

        output = model([inputs["input_ids"],
                        inputs["attention_mask"],
                        inputs["token_type_ids"]])

        start_logits = output[0]
        end_logits = output[1]

        sequence_length = inputs["input_ids"].shape[1]

        start_index = tf.math.argmax(
            start_logits[0, 1:sequence_length - 1]) + 1
        end_index = tf.math.argmax(end_logits[0, 1:sequence_length - 1]) + 1

        answer_tokens = inputs["input_ids"][0][start_index: end_index + 1]

        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=True)

        if not answer.strip():
            return None
        return answer
    except Exception as e:
        print(f"Error in retrieve_answer: {e}")
        return None


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


def question_answer(corpus_path):
    """Answers questions from multiple reference texts in a corpus.

    Args:
        corpus_path: The path to the corpus of reference documents.
    """

    while True:
        question = input("Q: ")
        question = question.strip().lower()

        if question in ("exit", "quit", "goodbye", "bye"):
            print("A: Goodbye")
            break

        document = semantic_search(corpus_path, question)

        if document:
            best_answer = retrieve_answer(question, document)
            if best_answer:
                print(f"A: {best_answer}")
            else:
                print("A: Sorry, I do not understand your question.")
        else:
            print("A: Sorry, I do not understand your question.")
