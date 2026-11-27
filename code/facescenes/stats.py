"""
Inference.

Two principles run through this module.

INFERENCE IS AT THE IMAGE LEVEL, NOT THE FOLD LEVEL. The 100 cross-validation
folds are overlapping resamples of the same images, so their spread reflects
resampling noise rather than sampling variability. Tests across folds are
badly anticonservative -- on pure-noise data they reject 23-96% of the time
instead of 5%. Every fold-wise quantity here is therefore averaged into a single
estimate, and the null distribution is built by permuting IMAGES and re-running
the whole fold pipeline, so the null reflects the complete analysis.

PERMUTATION MUST PRESERVE THE KIND OF OBJECT BEING COMPARED. Two nulls in
earlier versions of this analysis failed that test and are documented here so
the mistakes are not repeated:

  * Swapping the face-visible and face-masked feature values per image produces
    a MIXTURE predictor, which is systematically worse than either pure
    predictor, while the observed statistic compares two pure predictors. The
    resulting null was bimodal with no mass near zero. It is replaced by the
    image-level ANOVA in anova_2x2 below.
  * Swapping the two condition means per image makes both pseudo-conditions
    equivalent ~50/50 mixtures, yet the statistic still divided them by
    DIFFERENT reliabilities, placing the null at R * (1/rel_p - 1/rel_n) rather
    than zero. Two fixes are provided: recentre the null (recentre_null), or
    give both permuted mixtures the same reliability (see run_contrast_test.py).
    The observed statistic is unbiased -- each real condition is divided by its
    own correct reliability -- so only the NULL is adjusted.
"""

import numpy as np
from scipy import stats

from . import config as cfg
from . import encoding as E


def permutation_p(observed, null, tail="two-sided"):
    """(1 + #{null at least as extreme}) / (1 + #null).

    The add-one correction keeps p strictly positive: with n permutations the
    smallest attainable value is 1/(n+1), and reporting 0 would overstate the
    resolution of the test.

    A NaN observed value -- a feature that is not in the model, such as the
    untrained CNN in the unique analysis -- has no test, so its p is NaN. Without
    this guard every comparison with NaN is False, no null value counts as
    extreme, and the function would return the smallest possible p.
    """
    if np.isnan(observed):
        return np.nan
    null = np.asarray(null)
    if tail == "two-sided":
        k = np.sum(np.abs(null) >= abs(observed))
    elif tail == "greater":
        k = np.sum(null >= observed)
    elif tail == "less":
        k = np.sum(null <= observed)
    else:
        raise ValueError(tail)
    return (1 + k) / (1 + null.size)


def recentre_null(null):
    """Remove a known additive bias from a null distribution.

    Only valid where the statistic's expectation under H0 is zero. It is NOT
    valid for a statistic such as `reference - max(others)`, which is genuinely
    negative under H0 because the maximum of several noisy values is upward
    biased; recentring there would delete real structure.
    """
    null = np.asarray(null)
    return null - null.mean(0)


# ------------------------------------------------------------- paired tests
def dz_ci(t, n, alpha=0.05):
    """Confidence interval for Cohen's dz, dz +/- t_crit / sqrt(n)."""
    dz = t / np.sqrt(n)
    return stats.t.interval(1 - alpha, n - 1, dz, 1 / np.sqrt(n))


def one_sample_test(values):
    """Test a per-participant quantity against zero.

    Used for the within-participant regression coefficients of the onset-gaze
    control, where each participant contributes one slope and the group test
    asks whether those slopes differ from zero. Returns the same keys as
    paired_test where they mean the same thing, so both can be printed by the
    same code.
    """
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    t = stats.ttest_1samp(v, 0.0)
    return dict(n=int(v.size), mean=float(v.mean()),
                t=float(t.statistic), p=float(t.pvalue),
                dz=float(v.mean() / v.std(ddof=1)),
                ci=tuple(float(x) for x in stats.t.interval(
                    0.95, v.size - 1, v.mean(), v.std(ddof=1) / np.sqrt(v.size))))


def paired_test(a, b, log_a=None, log_b=None):
    """Paired comparison reported on the raw scale, tested on the log scale.

    `a` and `b` are per-participant means in the original units and are used for
    the descriptive statistics. `log_a` and `log_b` are per-participant means of
    the LOG-transformed trial values and, when supplied, carry the test.

    The distinction matters: averaging trials and then taking the log is not the
    same as averaging the logs, and the latter is what makes the reported
    geometric means and the t-test consistent with each other. Pass log_a/log_b
    as None for counts and percentages, which are tested on the raw scale --
    they are bounded and not multiplicative, so a log buys nothing.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    use_log = log_a is not None
    if use_log:
        log_a, log_b = np.asarray(log_a, float), np.asarray(log_b, float)
        ok = ~np.isnan(a) & ~np.isnan(b) & ~np.isnan(log_a) & ~np.isnan(log_b)
        x, y = log_a[ok], log_b[ok]
    else:
        ok = ~np.isnan(a) & ~np.isnan(b)
        x, y = a[ok], b[ok]
    a, b = a[ok], b[ok]
    d = x - y
    t = stats.ttest_rel(x, y)
    out = dict(n=int(d.size), mean_a=float(a.mean()), mean_b=float(b.mean()),
               sem_a=float(a.std(ddof=1) / np.sqrt(a.size)),
               sem_b=float(b.std(ddof=1) / np.sqrt(b.size)),
               t=float(t.statistic), p=float(t.pvalue),
               dz=float(d.mean() / d.std(ddof=1)),
               dz_ci=tuple(float(v) for v in dz_ci(t.statistic, d.size)),
               diff=float(d.mean()),
               diff_ci=tuple(float(v) for v in stats.t.interval(
                   0.95, d.size - 1, d.mean(), d.std(ddof=1) / np.sqrt(d.size))))
    if use_log:
        out["geo_a"] = float(np.exp(x.mean()))
        out["geo_b"] = float(np.exp(y.mean()))
        out["pct_change"] = float(100 * (np.exp(d.mean()) - 1))
    return out


# ------------------------------------------------------------- reliability
def split_half_reliability(values, images, rng, n_splits=2000, method="spearman"):
    """Spearman-Brown corrected split-half reliability of image-wise latencies.

    `values` is (n_participant, n_image) with NaN where a participant lacks data.
    Participants contributing to an image are split in two, the image mean is
    taken within each half, and the two vectors are correlated ACROSS IMAGES.
    Pass LOG latencies: the half means are then means of logs, the same
    quantity as data.image_means(log=True), which is what is modelled.

    method="spearman" matches how the encoding models are scored (Spearman) and
    noise-corrects predictivity. method="pearson" gives the ceiling for the
    Pearson R^2 of Fig. 1d/e, which is computed on the same log-scale means.
    """
    corr = {"spearman": lambda a, b: stats.spearmanr(a, b).statistic,
            "pearson": lambda a, b: stats.pearsonr(a, b).statistic}[method]
    rs = []
    for _ in range(n_splits):
        h1, h2 = [], []
        for j in images:
            idx = np.where(~np.isnan(values[:, j]))[0]
            if len(idx) < 4:
                continue
            rng.shuffle(idx)
            k = len(idx) // 2
            h1.append(np.nanmean(values[idx[:k], j]))
            h2.append(np.nanmean(values[idx[k:2 * k], j]))
        rs.append(corr(h1, h2))
    rs = np.asarray(rs)
    sb = 2 * rs / (1 + rs)                       # Spearman-Brown
    return float(sb.mean()), float(sb.std(ddof=1))


# ------------------------------------------------------------------- ANOVA
def anova_2x2(Y):
    """Two-way repeated-measures ANOVA, one observation per cell.

    Y is (n_unit, 2, 2) = replicate x factor A x factor B. The error term for
    each effect is that effect's interaction with the replicate, the standard
    within-subjects decomposition. Returns {effect: (F, df1, df2, p, eta2p)}.

    Verified against the equivalent paired t-tests: in a 2 x 2 within design
    every F equals t^2 of the matching paired contrast.
    """
    n, na, nb = Y.shape
    gm = Y.mean()
    unit = Y.mean((1, 2))
    am, bm = Y.mean((0, 2)), Y.mean((0, 1))
    SA, SB, AB = Y.mean(2), Y.mean(1), Y.mean(0)

    ss = {"A": n * nb * ((am - gm) ** 2).sum(),
          "Ae": nb * ((SA - unit[:, None] - am[None, :] + gm) ** 2).sum(),
          "B": n * na * ((bm - gm) ** 2).sum(),
          "Be": na * ((SB - unit[:, None] - bm[None, :] + gm) ** 2).sum(),
          "AxB": n * ((AB - am[:, None] - bm[None, :] + gm) ** 2).sum()}
    resid = (Y - SA[:, :, None] - SB[:, None, :] - AB[None, :, :]
             + unit[:, None, None] + am[None, :, None] + bm[None, None, :] - gm)
    ss["AxBe"] = (resid ** 2).sum()

    df2 = n - 1
    out = {}
    for k in ("A", "B", "AxB"):
        msd = ss[k + "e"] / df2
        F = ss[k] / msd if msd > 0 else np.nan
        out[k] = (float(F), 1, int(df2),
                  float(stats.f.sf(F, 1, df2)), float(F / (F + df2)))
    return out


def eta2p_ci(F, df1, df2, level=0.90):
    """Confidence interval for partial eta squared from an observed F.

    Inverts the noncentral F distribution for its noncentrality parameter
    lambda and converts each bound with eta2p = lambda / (lambda + df1 + df2 + 1)
    (Steiger, 2004). The 90% level is the one that matches a .05-level F test:
    the lower bound exceeds zero exactly when p < .05, because F is one-sided.
    """
    from scipy.optimize import brentq
    alpha = (1 - level) / 2

    def bound(q):
        """Noncentrality at which the observed F is the q-th quantile."""
        # lambda at which the observed F sits at the q-th quantile
        f = lambda lam: stats.ncf.cdf(F, df1, df2, lam) - q
        if f(0) < 0:                 # observed F below the central quantile
            return 0.0
        hi = 1.0
        while f(hi) > 0:
            hi *= 2
        return brentq(f, 0, hi)

    lo, up = bound(1 - alpha), bound(alpha)
    n = df1 + df2 + 1
    return lo / (lo + n), up / (up + n)


def spearman_ci(r, n, level=0.95):
    """Confidence interval for a Spearman correlation.

    Fisher z with the Bonett & Wright (2000) standard error for Spearman's rho,
    sqrt((1 + r^2 / 2) / (n - 3)), which is wider than the Pearson SE.
    """
    z = np.arctanh(r)
    se = np.sqrt((1 + r * r / 2) / (n - 3))
    q = stats.norm.ppf(1 - (1 - level) / 2)
    return float(np.tanh(z - q * se)), float(np.tanh(z + q * se))


def per_image_squared_error(z, y, splits, n_images):
    """Mean squared held-out prediction error for every image.

    Predictivity is defined over a SET of images and so has no per-image value,
    which is why a factorial test cannot be run on it directly. Squared error
    does decompose: each image is a test item in half the folds, and averaging
    its errors gives one value per image, making images the replicates of an
    ANOVA.
    """
    total = np.zeros(n_images)
    count = np.zeros(n_images)
    for train in splits:
        test = np.setdiff1d(np.arange(n_images), train)
        pred = E.ols_predict(z[train], y[train], z[test])
        total[test] += (y[test] - pred) ** 2
        count[test] += 1
    return total / count
