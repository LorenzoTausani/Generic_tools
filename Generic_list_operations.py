from typing import List, Callable, Any, Union, Dict

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


def multioption_prompt(opt_list: List[str], in_prompt: str) -> Union[str, List[str]]:
    """
    Prompt the user to choose from a list of options.

    Parameters:
    - opt_list: List of options.
    - in_prompt: Prompt message to display.

    Returns:
    - Either a single option (str) or a list of options (List[str]).
    """
    # Generate option list
    opt_prompt = '\n'.join([f'{i}: {opt}' for i, opt in enumerate(opt_list)])
    
    # Prompt user and evaluate input
    idx_answer = eval(input(f"{in_prompt}\n{opt_prompt}"))
    
    # Check if the answer is a list
    if isinstance(idx_answer, list):
        answer = [opt_list[idx] for idx in idx_answer]
    else:
        # If not a list, return the corresponding option
        answer = opt_list[idx_answer]

    return answer


def create_variable_dict(locals_or_globals: Dict, variables_list: list) -> Dict:
    """
    Create a dictionary associating each variable name to the variable in the local or global namespace.

    Parameters:
    - locals_or_globals: Dictionary-like object representing local or global namespace.
    - variables_list: List of variable names.

    Returns:
    A dictionary where keys are variable names and values are the corresponding variables.
    """
    variable_dict = {}
    for var in variables_list:
        if var in locals_or_globals:
            variable_dict[var] = locals_or_globals[var]
        else:
            print('\033[1mVariable {} not found\033[0m'.format(var))

    return variable_dict
