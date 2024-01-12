from typing import Any, Union
import re

def extract_numbers(input_string: str) -> str:
    """
    Estrae i numeri da una stringa.

    Parameters:
    - input_string: La stringa da cui estrarre i numeri.

    Returns:
    - output_string: La stringa risultante contenente solo i numeri.
    """
    return ''.join(char for char in input_string if char.isdigit())

def contains_character(s: str, pattern: str) -> bool:
    """
    Check if the given string contains at least one character that matches the specified pattern.

    Parameters:
    - s (str): The input string to check.
    - pattern (str): The pattern of characters to look for.

    Returns:
    - bool: True if the string contains at least one character matching the specified pattern, False otherwise.
    """
    return bool(re.search(pattern, s))

def exclude_chars(string: str, pattern: str) -> str:
    """
    Remove specified pattern from the input string.

    Parameters:
    - string (str): The input string.
    - pattern (str): The pattern to be excluded.

    Returns:
    - str: The string after removing the specified pattern.
    """
    return re.sub(pattern, '', string)

def substitute_character(input_string: str, old_char: str, new_char: str) -> str:
    """
    Substitute a particular character in a string with another character.

    Parameters:
    - input_string (str): The input string.
    - old_char (str): The character to be replaced.
    - new_char (str): The character to replace old_char.

    Returns:
    - str: The modified string.
    """
    return re.sub(re.escape(old_char), new_char, input_string) #non chiarissimo il ruolo di re.escape()


def standard_format_string(input_string: str) -> str:
    """
    Formatta una stringa con la prima lettera maiuscola e sostituisce gli spazi con underscore.

    Parameters:
    - input_string: La stringa da formattare.

    Returns:
    - formatted_string: La stringa risultante formattata.
    """
    words = input_string.split()
    formatted_words = [word.capitalize() for word in words]
    formatted_string = '_'.join(formatted_words)
    return formatted_string