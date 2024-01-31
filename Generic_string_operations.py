from typing import Any, Union, List
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
    Check if the given string contains at least one character that matches any of the specified patterns.

    Parameters:
    - s (str): The input string to check.
    - pattern (str or list): The pattern or list of characters to look for.

    Returns:
    - bool: True if the string contains at least one character matching any of the specified patterns, False otherwise.
    """
    if isinstance(pattern, list):
        # If pattern is a list, join the characters with '|' for the regex
        pattern = '|'.join(re.escape(char) for char in pattern)

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

def get_common_words_betw_strings(list_of_strings: List[str], delimiter: str = '-') -> List[str]:
    """
    Extracts common words between strings split by a given delimiter.

    Args:
    - list_of_strings (List[str]): List of strings to extract common words from.
    - delimiter (str, optional): Delimiter to split strings. Defaults to '-'.

    Returns:
    - List[str]: List of common words.
    """
    words_list = [s.split(delimiter) for s in list_of_strings]
    common_words = list(set(words_list[0]).intersection(*words_list[1:]))
    return common_words

def italic(string): #check this better
  return "\x1B[3m"+ string + "\x1B[0m"