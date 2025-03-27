#!/usr/bin/env python3
"""
Defines function that returns the list of schools with a specific topic
"""


def schools_by_topic(mongo_collection, topic):
    """
    Return a list of schools that offer a specific topic.

    Args:
        mongo_collection (pymongo.collection): The MongoDB collection object
        topic (str): The topic to search for

    Returns:
        list: A list of school documents that offer the specified topic.
    """
    schools = list(mongo_collection.find({"topics": topic}))
    return schools
