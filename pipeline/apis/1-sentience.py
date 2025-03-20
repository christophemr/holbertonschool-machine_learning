#!/usr/bin/env python3
"""
Defines methods to ping the Star Wars API and return the list of home planets
for all sentient species
"""

import requests


def sentientPlanets():
    """
    Returns a list of names of the home planets of all sentient species.

    Returns:
        list: A list of home planet names for sentient species.
    """
    planets = []
    next_page = "https://swapi.py4e.com/api/species/"

    while next_page:
        response = requests.get(next_page)
        data = response.json()

        # Iterate over the results and filter sentient species
        for species in data['results']:
            # Check if 'sentient' is in classification or designation
            if ('sentient' in species['classification'].lower() or
                    'sentient' in species['designation'].lower()):
                # Get the homeworld URL
                homeworld_url = species['homeworld']
                if homeworld_url:
                    homeworld_response = requests.get(homeworld_url)
                    homeworld_data = homeworld_response.json()
                    planets.append(homeworld_data['name'])

        # Check for the next page URL
        next_page = data['next']

    return planets
