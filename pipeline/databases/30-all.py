#!/usr/bin/env python3
"""
Defines function that lists all documents in MongoDB collection
"""


def list_all(mongo_collection):
    """
    List all documents in a MongoDB collection.

    Args:
        mongo_collection (pymongo.collection): The MongoDB collection object

    Returns:
        list: A list of all documents in the collection, or an empty list
        if no documents are found
    """
    # Find all documents in the collection
    documents = list(mongo_collection.find())

    # Return the list of documents
    return documents
