from typing import List, Callable, Any

def filter_list(input_list: List[Any], condition_func: Callable[[Any], bool]) -> List[Any]:
    """
    Filtra una lista in base a una condizione data da una funzione.

    Parameters:
    - input_list: Lista da filtrare.
    - condition_func: Funzione di condizione che restituisce True o False per ogni elemento.

    Returns:
    - filtered_list: Lista risultante dopo il filtro.
    """
    return [item for item in input_list if condition_func(item)]