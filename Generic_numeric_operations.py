from typing import Union, List, Tuple, Callable,Dict
from Generic_converters import *
from Generic_numpy_operations import *
import numpy as np
import torch
import pandas as pd
from itertools import combinations


def SEMf(data: Union[List[float], Tuple[float], np.ndarray], axis: int = 0) -> float:
    """
    Calcola lo standard error of the mean (SEM) per una sequenza di numeri.

    Parameters:
    - data: Sequenza di numeri (lista, tupla, array NumPy, ecc.).
    - axis: Asse lungo il quale calcolare la deviazione standard (valido solo per array NumPy).

    Returns:
    - sem: Standard error of the mean (SEM).
    """
    try:
        # Converte la sequenza in un array NumPy utilizzando la funzione convert_to_numpy
        data_array = convert_to_numpy(data)
        
        # Calcola la deviazione standard e il numero di campioni specificando l'asse se necessario
        std_dev = np.nanstd(data_array, axis=axis) if data_array.ndim > 1 else np.nanstd(data_array)
        sample_size = data_array.shape[axis]
        
        # Calcola lo standard error of the mean (SEM)
        sem = std_dev / np.sqrt(sample_size)
        
        return sem
    except Exception as e:
        print(f"Errore durante il calcolo dello standard error of the mean (SEM): {e}")
        return None
    

def perc_difference(array1: Union[List[float], np.ndarray, torch.Tensor, pd.DataFrame], array2: Union[List[float], np.ndarray, torch.Tensor, pd.DataFrame]) -> np.ndarray:
    """
    Calcola la differenza percentuale tra due array.

    Parameters:
    - array1 (Union[List[float], np.ndarray, torch.Tensor, pd.DataFrame]): Primo array di input.
    - array2 (Union[List[float], np.ndarray, torch.Tensor, pd.DataFrame]): Secondo array di input.

    Returns:
    - P_diff (np.ndarray): Array delle differenze percentuali.
    """
    array1 = convert_to_numpy(array1); array2 = convert_to_numpy(array2)
    return ((array1-array2)/array2)*100


def site_to_site_correlations(stim_data, phys_recording: np.ndarray, stimuli_of_interest : List[str], get_key_f: Callable[[Dict], List[str]],cells_of_interest: Union[List[int], np.ndarray] = [], n_it: int = 0) -> pd.DataFrame:
    """
    Calculate site-to-site correlations for a given set of stimuli.

    Parameters:
        stim_data: Stimulus data.
        phys_recording: Physiological recording data (cell x time).
        stimuli_of_interest: List of stimuli to analyze.
        cells_of_interest: List of cells to analyze.
        n_it: Index of the logical dictionary.

    Returns:
        pd.DataFrame: DataFrame containing site-to-site correlations.
    """
    if cells_of_interest == []: #if no cell is selected, then select them all
        cells_of_interest = np.arange(phys_recording.shape[0])
    else:
        cells_of_interest = np.sort(convert_to_numpy(cells_of_interest)) #otherwise, i sort the cells selected

    corr_site_couples = list(combinations(cells_of_interest, 2)); corr_df = pd.DataFrame({'Cell IDs': corr_site_couples}) #list all possible combinations between sites

    for stimulus in stimuli_of_interest: #for every stimulus listed...
        if stimulus == 'all': #if you want to see the correlation of all the trace, irrespective of timestamps...
            stim_avg_per_cell = phys_recording[cells_of_interest, :]
        else:
            if stimulus == 'stimuli' or stimulus == 'intertrial': #if you want to see the correlation among all stimuli, irrespective of their identity (or for their intertrials)...
                key_set, _ = get_key_f(stim_data.logical_dict[0])
                if stimulus == 'intertrial': 
                    key_set = ['gray ' + k for k in key_set]
                list_recs = [stim_data.get_stim_phys_recording(st, phys_recording, idx_logical_dict=n_it) for st in key_set]
                stimulus_phys_recording = np.concatenate(adjust_length_arrays(list_recs, dimensions_to_adjust = 2), axis=0)
            else: #if you want to see the correlation only for one specific type of stimulus...
                stimulus_phys_recording = stim_data.get_stim_phys_recording(stimulus, phys_recording, idx_logical_dict=n_it) #metti la possibilità di regolare durata stimolo?
            selected_stim_recs = stimulus_phys_recording[:, cells_of_interest, :]
            stim_avg_per_cell = np.mean(selected_stim_recs, axis=0)

        correlation_matrix = np.corrcoef(stim_avg_per_cell)
        upper_triangle_indices = np.triu_indices(correlation_matrix.shape[0], k=1)
        upper_triangle_corrs = correlation_matrix[upper_triangle_indices]
        corr_df["\u03C1-" + stimulus] = upper_triangle_corrs #"\u03C1" = ρ

    return corr_df

def get_indexes_in_combs(corr_df: pd.DataFrame, cell_indices: List[int]) -> List[int]:
    """
    Get the indices of elements in the DataFrame 'corr_df' where the 'Cell IDs' contain all integers in 'cell_indices'.

    Args:
        corr_df (pd.DataFrame): The DataFrame containing the column 'Cell IDs'.
        cell_indices (List[int]): The list of integers to search for.

    Returns:
        List[int]: The list of indices containing all integers in 'cell_indices'.
    """
    sel_idxs = [] # Initialize an empty list to store indices
    
    for idx, val in enumerate(corr_df['Cell IDs']): # Iterate over each element in the Series
        # Check if all integers in the tuple match those in cell_indices
        if all(num in cell_indices for num in val):
            sel_idxs.append(idx) # If match found, append the index to the list

    return sel_idxs