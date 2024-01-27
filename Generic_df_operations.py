import pandas as pd
from pandas.core.frame import DataFrame

def round_numeric_columns(df: DataFrame, n=2) -> DataFrame:
    """
    Round numeric columns in a DataFrame to n decimals after the comma.

    Parameters:
    - df: DataFrame to round numeric columns.
    - n: the number of digits to round after the comma

    Returns:
    A new DataFrame with numeric columns rounded to n decimals after the comma.
    """
    return df.apply(lambda x: round(x, n) if pd.api.types.is_numeric_dtype(x) else x)
