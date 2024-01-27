from Generic_string_operations import *

def sort_by_numeric_part(item: str) -> float:
    """
    Sort function to sort elements based on their numeric part.
    NOTE: it was developed specifically for oriented gratings sorting, based on both orientation
    and direction (+/-)

    Parameters:
    - item: A string representing an element to be sorted.

    Returns:
    A float representing the numeric part of the element.
    """
    # Extract the numeric part of each element
    number_part = int(extract_numbers(item))

    # Adjust the sorting based on the last character (+ or -)
    if item[-1] == '-':
        return number_part
    else:
        return number_part + 0.5  # Add 0.5 to treat '-' and '+' as close