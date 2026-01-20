import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestIndPower

def normality(x):
    s = stats.shapiro(x)[1]
    d = stats.normaltest(x)[1]
    return s, d

def fit_gaussian(x):
    mu, sigma = stats.norm.fit(x)
    return float(mu), float(sigma)

def ci_mean(x, normal=True):
    x = np.asarray(x)
    n = x.shape[0]
    if normal:
        mu, sigma = fit_gaussian(x)
        se = sigma / np.sqrt(n)
        return mu - 1.96*se, mu + 1.96*se
    bs = np.random.default_rng(42).choice(x, size=(10000, n), replace=True).mean(axis=1)
    return np.quantile(bs, 0.025), np.quantile(bs, 0.975)

def compare(a, b):
    sa = stats.shapiro(a)[1] > 0.05
    sb = stats.shapiro(b)[1] > 0.05
    if sa and sb:
        t, p = stats.ttest_ind(a, b, equal_var=False)
    else:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    ma = float(np.mean(a)); mb = float(np.mean(b))
    sda = float(np.std(a, ddof=1)); sdb = float(np.std(b, ddof=1))
    na = len(a); nb = len(b)
    sp = np.sqrt(((na-1)*sda**2 + (nb-1)*sdb**2) / (na+nb-2))
    d = (ma - mb) / sp if sp > 0 else 0.0
    return p, d

def power(effect_size, n_opt, n_base):
    return TTestIndPower().solve_power(effect_size=abs(effect_size), nobs1=n_opt, ratio=n_base/n_opt, alpha=0.05, alternative="two-sided")
