#!/usr/bin/env python3
"""
Defines methods to ping the Star Wars API and return the list of ships
that can hold a given number of passengers
"""

import requests


def availableShips(passengerCount):
    """
    Returns a list of ships that can hold a given number of passengers.

    Args:
        passengerCount (int): The number of passengers to accommodate.

    Returns:
        list: List of ship names that can hold the given number of passengers.
    """
    ships = []
    next_page = "https://swapi-api.hbtn.io/api/starships/"

    while next_page:
        response = requests.get(next_page)
        data = response.json()

        # Iterate over the results and filter ships by passenger capacity
        for ship in data['results']:
            # Check if the passengers field is a valid number
            if ship['passengers'].isdigit():
                if int(ship['passengers']) >= passengerCount:
                    ships.append(ship['name'])

        # Check for the next page URL
        next_page = data['next']

    return ships
