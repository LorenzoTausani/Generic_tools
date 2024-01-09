from typing import Union, List, Tuple
from Generic_converters import *
import numpy as np
import torch
import pandas as pd

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