import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import comb
import numpy as np
import pandas as pd
from typing import Literal,Union, List

from Generic_stats import *

def set_default_matplotlib_params(side: float = 15, shape: Literal['square', 'rect_tall', 'rect_wide'] = 'square', sns_params=False) -> None:
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
    if sns_params:
        sns_params_dict = {}
        sns_params_dict['medianprops']={"color":median_c,"linewidth":median_lw}; 
        sns_params_dict['whiskerprops']={"color":box_c,"linewidth":box_lw}; 
        sns_params_dict['boxprops']={"edgecolor":box_c,"linewidth":box_lw}
        return params,sns_params_dict

    return params

def draw_2test_lines(results: Dict[str, float], col_idxs: List[float], y_position: float, next_segment: float) -> float:
  """
  Annotate a box plot with horizontal lines to underline statistical significant paired statistical test comparison.

  Parameters:
  - results (Dict[str, float]): Dictionary containing statistical test results (e.g., p-value).
  - col_idxs (List[float]): coordinates of the columns to be compared
  - y_position (float): Current vertical position on the plot.
  - next_segment (float): Vertical distance to move to the next segment.

  Returns:
  - float: Updated vertical position after drawing lines and annotations.
  """
  if results['p_value'] < 0.05: #we add the significance segment only if the test result is significant
    x_position = (col_idxs[0] + col_idxs[1] + 2) / 2.0 #compute average horizontal position between the columns
    y_position = y_position+next_segment #compute the vertical position of the significance segment
    plt.plot([col_idxs[0] + 1, col_idxs[1] + 1], [y_position, y_position], color='black') #draw the significance segment
    font_size = plt.rcParams['xtick.labelsize'] / 3.0 #compute the fontsize to be used for the p-value
    plt.text(x_position, y_position + next_segment/3, "p = {:.2e}".format(results["p_value"]),
            ha='center', va='center', fontsize=font_size) #add the p-value text above the significance segment
    return y_position

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
              col1 = selected_columns[i]; col2 = selected_columns[j]; col_idxs = [i,j]
              results = two_sample_test(df[col1], df[col2])
              y_position = draw_2test_lines(results, col_idxs, y_position, next_segment)

    if omitplot==False:
      plt.title(title)
      #plt.savefig(session_name+'Fluorescence_periods_comparison.png') da mettere
      plt.show()


def paired_boxplot(df_pre: pd.DataFrame, df_post: pd.DataFrame, vars_of_interest: List[str],
                   Pre_Post_lbls: Dict[str, str] = {'Pre': 'green', 'Post': 'purple'},
                   y_lim: Optional[List[float]] = None) -> None:
    """
    Create a paired boxplot comparing two dataframes for the variables vars_of_interest.

    Parameters:
    - df_pre (pd.DataFrame): The dataframe for the 'Pre' condition.
    - df_post (pd.DataFrame): The dataframe for the 'Post' condition.
    - vars_of_interest (List[str]): List of variables to be compared.
    - Pre_Post_lbls (Dict[str, str]): Labels and colors for 'Pre' and 'Post' conditions.
    - y_lim (Optional[List[float]]): Optional y-axis limits.

    Returns:
    - None
    """
    #some graphic settings
    grouping_name = 'Treatment'; x_lbl = 'Metric'; palette = [Pre_Post_lbls['Pre'], Pre_Post_lbls['Post']]
    _,sns_params = set_default_matplotlib_params(sns_params=True)
    #combining data in a single dataframe
    pre = df_pre[vars_of_interest]; post = df_post[vars_of_interest]; Pre_Post_keys = [key for key in Pre_Post_lbls]
    pre[grouping_name] = Pre_Post_keys[0]; post[grouping_name] = Pre_Post_keys[1]
    combined_data = pd.concat([pre, post], ignore_index=True)
    #melted_data is prepared for seaborn-specific processing
    if isinstance(vars_of_interest,list):
        v_name = '/'.join(vars_of_interest)
    else:
        v_name = vars_of_interest
    melted_data = pd.melt(combined_data, id_vars=[grouping_name], value_vars=vars_of_interest, var_name=x_lbl, value_name=v_name)
    sns.boxplot(y=v_name, x=x_lbl, hue=grouping_name, data=melted_data,gap=.1, medianprops =sns_params['medianprops'], 
                    boxprops=sns_params['boxprops'], whiskerprops=sns_params['whiskerprops'], capprops =sns_params['whiskerprops'],
                    palette=palette)
    plt.legend(title=grouping_name, bbox_to_anchor=(1.05, 0.6), loc='upper left')
    if not(y_lim==[]):
        plt.ylim(y_lim)
        y_position = y_lim[-1]
    else:
        y_position = melted_data.max(numeric_only=True).max()
    next_segment = 0.1*y_position
    col_idxs = [-1.2,-0.8]
    for i in range(len(vars_of_interest)):
        results = two_sample_test(pre[vars_of_interest[i]], post[vars_of_interest[i]])
        y_position = draw_2test_lines(results,col_idxs, y_position, next_segment)
        col_idxs = [c+1 for c in col_idxs]
    plt.show()



