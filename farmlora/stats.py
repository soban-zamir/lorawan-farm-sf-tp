"""Statistics: means over farms, 95 % confidence intervals and paired t-tests."""
import numpy as np
from scipy import stats


def ci95(x):
    """Half-width of the 95 % confidence interval of the mean (t distribution), ignoring NaNs."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def mci(v, d=2):
    """'mean ± half-width' string."""
    return f'{np.nanmean(v):7.{d}f} ± {ci95(v):5.{d}f}'


def paired(x, y):
    """Paired comparison of x against y (same farms): mean difference, its 95 % CI half-width,
    the paired t-test p-value and the difference in % of mean(y)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = ~np.isnan(x) & ~np.isnan(y)
    if ok.sum() < 2:
        return np.nan, np.nan, np.nan, np.nan
    d = x[ok] - y[ok]
    return d.mean(), ci95(d), stats.ttest_rel(x[ok], y[ok]).pvalue, 100 * d.mean() / y[ok].mean()
