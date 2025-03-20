#!/usr/bin/env python3
"""
Defines methods to ping the Star Wars API and return whether the 'Death Star'
is among the ships that can hold a given number of passengers.
"""

import requests


def availableShips(passengerCount):
    """
    Checks a list of  the ships that can hold a given number of passengers.

    Args:
        passengerCount (int): The number of passengers to accommodate.

    Returns:
        str: "OK" if 'Death Star' is found,
        otherwise "Ships not found: ['Death Star']".
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

    # Check if 'Death Star' is in the list of ships
    if 'Death Star' in ships:
        return "OK"
    else:
        return "Ships not found: ['Death Star']"


if __name__ == '__main__':
    # Example usage: Check for ships that can hold at least 1 passenger
    result = availableShips(1)
    print(result)
