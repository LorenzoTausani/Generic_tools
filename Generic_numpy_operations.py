from typing import List, Union
import numpy as np

def adjust_length_arrays(arrays_list: List[np.ndarray], dimensions_to_adjust: Union[int, List[int]]) -> List[np.ndarray]:
    """
    Adjusts the length of numpy arrays in a list along specified dimensions (especially useful for concat operations).

    Args:
        arrays_list (List[np.ndarray]): A list of numpy arrays.
        dimensions_to_adjust (Union[int, List[int]], optional): The dimension indices to adjust.

    Returns:
        List[np.ndarray]: The list of numpy arrays with adjusted lengths.
    """
    if isinstance(dimensions_to_adjust, int):
        dimensions_to_adjust = [dimensions_to_adjust] # Convert single dimension to list if provided

    min_dimension_lengths = [np.min([arr.shape[dim] for arr in arrays_list]) for dim in dimensions_to_adjust] # Find the minimum value of the specified dimensions
   
    for idx, arr in enumerate(arrays_list): # Iterate over the arrays in the list
        # Check if the dimensions lengths match the minimum
        for dim, min_dim_length in zip(dimensions_to_adjust, min_dimension_lengths):
            if arr.shape[dim] != min_dim_length: # If it's longer, cut it to match the minimum dimension length
                slices = [slice(None)] * len(arr.shape) #slice(None) = null slicer (i.e. takes all)
                slices[dim] = slice(int(min_dim_length)) #slice object to cut till min_dim_length
                arrays_list[idx] = arr[tuple(slices)] #do the cutting on arr
                arr = arrays_list[idx]
    return arrays_list