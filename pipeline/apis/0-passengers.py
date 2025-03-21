#!/usr/bin/env python3
"""
Defines methods to ping the Star Wars API and return whether the 'Death Star'
is among the ships that can hold a given number of passengers.
"""

import requests


def availableShips(passengerCount):
    """
    Checks a list of the ships that can hold a given number of passengers.

    Args:
        passengerCount (int): The number of passengers to accommodate.

    Returns:
        str: "OK" if 'Death Star' is found,
        otherwise "Ships not found: <list>".
    """
    ships = []
    next_page = "https://swapi-api.hbtn.io/api/starships/"

    while next_page:
        response = requests.get(next_page)
        if response.status_code != 200:
            break
        data = response.json()

        # Iterate over the results and filter ships by passenger capacity
        for ship in data['results', []]:
            passengers = ship.get("passengers", "0").replace(",", "")
            if passengers.isdigit() and int(passengers) >= passengerCount:
                ships.append(ship.get("name"))

        # Check for the next page URL
        next_page = data.get("next")

    return ships
