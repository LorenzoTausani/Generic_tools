from typing import Any, Union

def extract_numbers(input_string: str) -> str:
    """
    Estrae i numeri da una stringa.

    Parameters:
    - input_string: La stringa da cui estrarre i numeri.

    Returns:
    - output_string: La stringa risultante contenente solo i numeri.
    """
    return ''.join(char for char in input_string if char.isdigit())

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