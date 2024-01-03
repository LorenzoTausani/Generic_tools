from typing import Union, List, Tuple
from Generic_converters import *
import numpy as np

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