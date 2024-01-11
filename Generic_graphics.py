import matplotlib.pyplot as plt
from scipy.special import comb
import numpy as np
import pandas as pd
from typing import Literal,Union, List

from Generic_stats import *

def set_default_matplotlib_params(side: float = 15, shape: Literal['square', 'rect_tall', 'rect_wide'] = 'square') -> None:
    """
    Set default Matplotlib parameters for better visualizations.

    Parameters:
    - side: Length of one side of the figure (default is 15).
    - shape: Shape of the figure ('square', 'rect_tall', or 'rect_wide', default is 'square').

    Returns:
    - None
    """
    if shape == 'square':
        other_side = side
    elif shape == 'rect_wide':
        other_side = int(side * (2/3))
    elif shape == 'rect_tall':
        other_side = int(side * (3/2))
    else:
        raise ValueError("Invalid shape. Use 'square', 'rect_tall', or 'rect_wide'.")
    writing_sz = 50; standard_lw = 4
    box_lw = 3; box_c = 'black'; median_lw = 4; median_c = 'red'
    params = {
        'figure.figsize': (side, other_side),
        'font.size': writing_sz,
        'axes.labelsize': writing_sz,
        'axes.titlesize': writing_sz,
        'xtick.labelsize': writing_sz,
        'ytick.labelsize': writing_sz,
        'legend.fontsize': writing_sz,
        'axes.grid': False,
        'grid.alpha': 0.5,
        'lines.linewidth': standard_lw,
        'lines.markersize': 12,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
        'errorbar.capsize': standard_lw,
        'boxplot.boxprops.linewidth': box_lw,
        'boxplot.boxprops.color': box_c,
        'boxplot.whiskerprops.linewidth': box_lw,
        'boxplot.whiskerprops.color': box_c,
        'boxplot.medianprops.linewidth': median_lw,
        'boxplot.medianprops.color': median_c,
        'boxplot.capprops.linewidth': box_lw,
        'boxplot.capprops.color': box_c
    }

    plt.rcParams.update(params)


def custom_boxplot(df: pd.DataFrame,
                   selected_columns: List[str],
                   xlbl: str = 'Condition',
                   ylbl: str = 'Fluorescence',
                   test2: bool = True,
                   omitplot: bool = False,
                   title: str = '') -> None:
    """
    Crea un boxplot per un DataFrame con la possibilità di aggiungere test statistici tra le colonne.

    Parameters:
    - df (pd.DataFrame): Il DataFrame contenente i dati.
    - selected_columns (List[str]): Lista delle colonne da considerare nel boxplot.
    - xlbl (str): Etichetta dell'asse x.
    - ylbl (str): Etichetta dell'asse y.
    - test2 (bool): Se True, calcola i p-value e aggiunge segmenti significativi tra le colonne.
    - omitplot (bool): Se False, mostra il boxplot.
    - title (str): Titolo del boxplot.

    Returns:
    - None: La funzione mostra il boxplot o lo salva a seconda dei parametri forniti.
    """
    set_default_matplotlib_params(side=15, shape='rect_wide')
    boxplot = plt.boxplot(df[selected_columns], labels=selected_columns)
    plt.xlabel(xlbl)
    plt.ylabel(ylbl)
    y_position = df.max().max() 
    next_segment = 0.1*y_position
    min_y = df.min().min()-next_segment
    number_of_stat_comps = comb(len(selected_columns), 2)
    y_lim = y_position+(number_of_stat_comps+1)*next_segment
    plt.ylim([min_y,y_lim])

    if test2: # Calcola i p-value e aggiungi i segmenti significativi
      for i in range(len(selected_columns)):
          for j in range(i + 1, len(selected_columns)):
              col1 = selected_columns[i]
              col2 = selected_columns[j]
              results = two_sample_test(df[col1], df[col2])
              
              # Aggiungi segmento solo se il p-value è significativo
              if results['p_value'] < 0.05:
                  # Calcola la posizione orizzontale media tra le colonne
                  x_position = (i + j + 2) / 2.0

                  # Calcola la posizione verticale in base al massimo valore tra le colonne
                  y_position = y_position+next_segment
    
                  # Aggiungi il segmento orizzontale
                  plt.plot([i + 1, j + 1], [y_position, y_position], color='black')

                  # Calcola la dimensione del carattere
                  font_size = plt.rcParams['xtick.labelsize'] / 3.0
                  # Aggiungi il testo con il p-value sopra il segmento
                  plt.text(x_position, y_position + next_segment/3, "p = {:.2e}".format(results["p_value"]),
                          ha='center', va='center', fontsize=font_size)
    if omitplot==False:
      plt.title(title)
      #plt.savefig(session_name+'Fluorescence_periods_comparison.png') da mettere
      plt.show()