#Repository funzioni generiche - convertitori
from typing import Any, Union
import numpy as np
import torch
import pandas as pd

def convert_to_numpy(data: Union[list, tuple, np.ndarray, torch.Tensor, pd.DataFrame]) -> np.ndarray:
    """
    Converte un qualsiasi dato in un array NumPy.

    Parameters:
    - data: Il dato da convertire (può essere una lista, una tupla, un array NumPy, un tensore PyTorch o un DataFrame pandas).

    Returns:
    - numpy_array: L'array NumPy risultante.
    """
    try:
        if isinstance(data, pd.DataFrame):
            numpy_array = data.to_numpy()
        elif isinstance(data, torch.Tensor):
            numpy_array = data.numpy()
        else:
            numpy_array = np.array(data)
        return numpy_array
    except Exception as e:
        print(f"Errore durante la conversione in NumPy array: {e}")
        return None

