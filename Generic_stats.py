import numpy as np
from scipy.stats import shapiro, ttest_rel, ttest_ind, wilcoxon, mannwhitneyu

from Generic_converters import *

def two_sample_test(vector1, vector2, paired=True, alpha=0.05, small_sample_size=20):
    """
    Perform a two-sample statistical test, considering normality assumptions.

    Parameters:
    - vector1 (array-like): First vector of values.
    - vector2 (array-like): Second vector of values.
    - paired (bool): If True, perform a paired test; if False, perform an unpaired test. Default is True.
    - alpha (float): Significance level for normality test. Default is 0.05.
    - small_sample_size (int): Sample size threshold for using non-parametric tests. Default is 20.

    Returns:
    - test_name(str): test_name of the statistical test ('parametric' or 'non-parametric').
    - p_value (float): P-value of the chosen test.
    """
    vector1 = convert_to_numpy(vector1); vector2 = convert_to_numpy(vector2)
    n1, n2 = len(vector1), len(vector2)
    if n1 != n2:
        paired = False

    results_two_sample_test = {}
    # Shapiro-Wilk test for normality
    _, p_value_normal1 = shapiro(vector1)
    _, p_value_normal2 = shapiro(vector2)
    results_two_sample_test['Normality test'] = [p_value_normal1,p_value_normal2]

    # Choose the appropriate test based on normality and sample size
    if paired:        
        if p_value_normal1 > alpha and p_value_normal2 > alpha:
            # Both vectors are approximately normally distributed
            if n1 >= small_sample_size:
                _, p_value = ttest_rel(vector1, vector2)
                test_name = 'ttest_rel'
            else:
                _, p_value = wilcoxon(vector1, vector2)
                test_name= 'wilcoxon'
        else:
            # At least one vector is not normally distributed
            _, p_value = wilcoxon(vector1, vector2)
            test_name= 'wilcoxon'
    else:
        if p_value_normal1 > alpha and p_value_normal2 > alpha:
            # Both vectors are approximately normally distributed
            if min(n1, n2) >= small_sample_size:
                _, p_value = ttest_ind(vector1, vector2, equal_var=True)
                test_name= 'ttest_ind'
            else:
                _, p_value = mannwhitneyu(vector1, vector2)
                test_name= 'mannwhitneyu'
        else:
            # At least one vector is not normally distributed
            _, p_value = mannwhitneyu(vector1, vector2)
            test_name= 'mannwhitneyu'

    print(f"Test name: {test_name}\nP-value: {p_value}")
    results_two_sample_test['Test performed'] = test_name 
    results_two_sample_test['p_value'] = p_value 
    return results_two_sample_test

