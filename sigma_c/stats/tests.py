import numpy as np
from scipy import stats

def pool_adjacent_violators(y, weights=None, increasing=True):
    """
    Pool-Adjacent-Violators Algorithm for isotonic regression.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    
    if weights is None:
        weights = np.ones(n)
    else:
        weights = np.asarray(weights, dtype=float)
    
    if not increasing:
        y = -y
    
    fitted = y.copy()
    w = weights.copy()
    
    i = 0
    while i < n - 1:
        if fitted[i] > fitted[i + 1]:
            total_weight = w[i] + w[i + 1]
            pooled_value = (fitted[i] * w[i] + fitted[i + 1] * w[i + 1]) / total_weight
            
            fitted[i] = pooled_value
            fitted[i + 1] = pooled_value
            w[i] = total_weight
            w[i + 1] = 0
            
            j = i - 1
            while j >= 0 and fitted[j] > fitted[j + 1]:
                total_weight = w[j] + w[j + 1]
                pooled_value = (fitted[j] * w[j] + fitted[j + 1] * w[j + 1]) / total_weight
                fitted[j] = pooled_value
                fitted[j + 1] = pooled_value
                w[j] = total_weight
                w[j + 1] = 0
                j -= 1
            
            i += 1
        else:
            i += 1
            
    for i in range(n):
        if w[i] == 0:
            j = i - 1
            while j >= 0 and w[j] == 0:
                j -= 1
            if j >= 0:
                fitted[i] = fitted[j]
                
    if not increasing:
        fitted = -fitted
        
    return fitted

def jonckheere_terpstra_test(values, groups):
    """
    Jonckheere-Terpstra test for ordered alternatives.
    """
    n_groups = len(values)
    S = 0
    for i in range(n_groups):
        for j in range(i+1, n_groups):
            if values[j] < values[i]:
                S += 1
            elif values[j] > values[i]:
                S -= 1
                
    n = n_groups
    var_S = n * (n - 1) * (2 * n + 5) / 18
    
    if var_S == 0:
        z = 0
        p_value = 0.5
    else:
        if S > 0:
            z = (S - 1) / np.sqrt(var_S)
        elif S < 0:
            z = (S + 1) / np.sqrt(var_S)
        else:
            z = 0
        p_value = stats.norm.sf(abs(z))
        
    return {
        'statistic': S,
        'z_score': z,
        'p_value': p_value,
        'significant': p_value < 0.05
    }

def isotonic_regression_with_ci(x, y, n_bootstrap=1000, increasing=False):
    """
    Isotonic regression with bootstrap confidence intervals.
    """
    iso_fit = pool_adjacent_violators(y, increasing=increasing)
    
    n_points = len(x)
    bootstrap_fits = np.zeros((n_bootstrap, n_points))
    
    for i in range(n_bootstrap):
        idx = np.random.choice(n_points, n_points, replace=True)
        boot_y = y[idx]
        # Sort x to maintain order assumption for PAV
        # Note: This is a simplified bootstrap for isotonic regression
        # strictly speaking we should resample (x,y) pairs and re-sort by x
        
        # Correct approach: resample residuals or pairs
        # Here we use simple pair resampling but we must sort by x for PAV
        sorted_idx = np.argsort(x[idx])
        boot_y_sorted = boot_y[sorted_idx]
        
        try:
            boot_fit = pool_adjacent_violators(boot_y_sorted, increasing=increasing)
            bootstrap_fits[i] = boot_fit
        except:
            bootstrap_fits[i] = iso_fit
            
    ci_lower = np.percentile(bootstrap_fits, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_fits, 97.5, axis=0)
    
    return {
        'fitted': iso_fit,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper
    }
