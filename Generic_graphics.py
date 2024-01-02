import matplotlib.pyplot as plt
from typing import Literal

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
    elif shape == 'rect_tall':
        other_side = int(side * (2/3))
    elif shape == 'rect_wide':
        other_side = int(side * (3/2))
    else:
        raise ValueError("Invalid shape. Use 'square', 'rect_tall', or 'rect_wide'.")

    params = {
        'figure.figsize': (side, other_side),
        'font.size': 50,
        'axes.labelsize': 50,
        'axes.titlesize': 50,
        'xtick.labelsize': 50,
        'ytick.labelsize': 50,
        'legend.fontsize': 50,
        'axes.grid': False,
        'grid.alpha': 0.5,
        'lines.linewidth': 4,
        'lines.markersize': 12,
        'xtick.major.pad': 5,
        'ytick.major.pad': 5,
        'errorbar.capsize': 4,
        'errorbar.elinewidth': 5
    }

    plt.rcParams.update(params)
