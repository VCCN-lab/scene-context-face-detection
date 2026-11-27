"""Unit tests for facescenes.stats, on synthetic data with known answers."""

import warnings

import numpy as np
import pytest
from scipy import stats

from facescenes import encoding as E
from facescenes import nan
from facescenes import stats as S


# ------------------------------------------------------------ permutation_p
class TestPermutationP:
    def test_add_one_floor(self):
        """An observed value beyond the whole null gives p = 1 / (n + 1)."""
        # observed beyond every null value: the smallest attainable p, never 0
        assert S.permutation_p(10, np.zeros(99), "greater") == pytest.approx(1 / 100)

    def test_tails(self):
        """greater, less and two-sided tails count the matching null values."""
        null = np.arange(-50, 50)                     # 100 values
        assert S.permutation_p(45, null, "greater") == pytest.approx((1 + 5) / 101)
        assert S.permutation_p(-45, null, "less") == pytest.approx((1 + 6) / 101)
        # two-sided counts |null| >= 45: -50..-45 and 45..49
        assert S.permutation_p(45, null) == pytest.approx((1 + 11) / 101)

    def test_ties_count_as_extreme(self):
        """Null values equal to the observed one count as extreme."""
        assert S.permutation_p(1.0, np.ones(9), "greater") == pytest.approx(1.0)

    def test_nan_observed_gives_nan(self):
        """A NaN observed value gives a NaN p."""
        # A feature not in the model has no test. Before the fix this returned
        # the minimum p, 1/(n+1), because every comparison with NaN is False.
        assert np.isnan(S.permutation_p(np.nan, np.full(100, np.nan), "greater"))
        assert np.isnan(S.permutation_p(np.nan, np.zeros(100)))

    def test_unknown_tail(self):
        """An unknown tail raises ValueError."""
        with pytest.raises(ValueError):
            S.permutation_p(0, np.zeros(3), "sideways")

    def test_uniform_under_null(self):
        """Under the null, p < .05 occurs in about 5% of tests."""
        rng = np.random.default_rng(0)
        ps = [S.permutation_p(rng.normal(), rng.normal(size=199), "greater")
              for _ in range(2000)]
        assert np.mean(np.array(ps) < .05) == pytest.approx(.05, abs=.015)


def test_recentre_null():
    """A recentred null has zero mean in every column."""
    null = np.random.default_rng(1).normal(3, 1, size=(500, 4))
    assert np.allclose(S.recentre_null(null).mean(0), 0)


# ------------------------------------------------------------ paired tests
class TestPairedTest:
    def setup_method(self):
        """Paired lognormal samples with b about 5% above a."""
        rng = np.random.default_rng(2)
        self.a = rng.lognormal(5.3, .2, 30)
        self.b = self.a * rng.lognormal(.05, .05, 30)

    def test_raw_matches_scipy(self):
        """On the raw scale t, p, n and means match scipy's paired t-test."""
        r = S.paired_test(self.a, self.b)
        t = stats.ttest_rel(self.a, self.b)
        assert r["t"] == pytest.approx(t.statistic)
        assert r["p"] == pytest.approx(t.pvalue)
        assert r["n"] == 30
        assert r["mean_a"] == pytest.approx(self.a.mean())
        assert "geo_a" not in r

    def test_log_scale(self):
        """With log values the test runs on logs, with geometric means and
        percent change, and descriptives stay raw."""
        la, lb = np.log(self.a), np.log(self.b)
        r = S.paired_test(self.a, self.b, la, lb)
        assert r["t"] == pytest.approx(stats.ttest_rel(la, lb).statistic)
        assert r["geo_a"] == pytest.approx(np.exp(la.mean()))
        assert r["pct_change"] == pytest.approx(100 * (np.exp((la - lb).mean()) - 1))
        # descriptives stay on the raw scale
        assert r["mean_a"] == pytest.approx(self.a.mean())

    def test_dz_and_cis(self):
        """dz is the mean over SD of the differences, the CIs contain their
        estimates, and the difference CI excludes 0 iff p < .05."""
        r = S.paired_test(self.a, self.b)
        d = self.a - self.b
        assert r["dz"] == pytest.approx(d.mean() / d.std(ddof=1))
        assert r["dz_ci"][0] < r["dz"] < r["dz_ci"][1]
        assert r["diff_ci"][0] < r["diff"] < r["diff_ci"][1]
        # the CI excludes zero exactly when p < .05
        assert (r["diff_ci"][1] < 0 or r["diff_ci"][0] > 0) == (r["p"] < .05)

    def test_nan_pairs_dropped(self):
        """Pairs with a NaN on either side are dropped from the test."""
        a, b = self.a.copy(), self.b.copy()
        a[0], b[5] = np.nan, np.nan
        r = S.paired_test(a, b)
        assert r["n"] == 28
        keep = np.ones(30, bool); keep[[0, 5]] = False
        assert r["t"] == pytest.approx(stats.ttest_rel(self.a[keep], self.b[keep]).statistic)


def test_one_sample_test():
    """one_sample_test matches scipy, drops NaNs and brackets the mean."""
    v = np.random.default_rng(3).normal(1, 2, 40)
    v[3] = np.nan
    r = S.one_sample_test(v)
    t = stats.ttest_1samp(v[~np.isnan(v)], 0)
    assert r["n"] == 39
    assert r["t"] == pytest.approx(t.statistic)
    assert r["p"] == pytest.approx(t.pvalue)
    assert r["ci"][0] < r["mean"] < r["ci"][1]


def test_dz_ci_width():
    """The dz CI is centred on t / sqrt(n)."""
    lo, hi = S.dz_ci(t=4.0, n=38)
    assert (lo + hi) / 2 == pytest.approx(4 / np.sqrt(38))


# ------------------------------------------------------------- reliability
class TestSplitHalfReliability:
    def make(self, noise, n_sub=40, n_img=60, seed=4):
        """Participant x image values: a true image effect plus Gaussian noise
        of SD `noise`."""
        rng = np.random.default_rng(seed)
        true = rng.normal(size=n_img)
        return true[None, :] + noise * rng.normal(size=(n_sub, n_img))

    def test_noise_free_is_one(self):
        """Noise-free data have a reliability of one."""
        v = self.make(noise=1e-9)
        m, _ = S.split_half_reliability(v, range(60), np.random.default_rng(0), 50)
        assert m == pytest.approx(1, abs=1e-6)

    def test_pure_noise_is_near_zero(self):
        """Data without an image effect have a reliability near zero."""
        rng = np.random.default_rng(5)
        v = rng.normal(size=(40, 60))
        m, _ = S.split_half_reliability(v, range(60), rng, 200)
        assert abs(m) < .15

    def test_matches_spearman_brown_theory(self):
        """Pearson reliability matches the Spearman-Brown prediction."""
        # signal var 1, noise var 16, 20 per half: r_half = 1/(1+16/20)
        v = self.make(noise=4)
        m, _ = S.split_half_reliability(v, range(60), np.random.default_rng(1),
                                        300, method="pearson")
        r_half = 1 / (1 + 16 / 20)
        assert m == pytest.approx(2 * r_half / (1 + r_half), abs=.08)

    def test_nans_and_sparse_images(self):
        """NaNs, empty images and images with few observers still give a
        finite, high reliability."""
        v = self.make(noise=.5)
        v[:20, :] = np.nan                     # half the participants per image
        v[:, 0] = np.nan                       # an image with no data is skipped
        v[:37, 1] = np.nan                     # fewer than 4 observers: skipped
        m, _ = S.split_half_reliability(v, range(60), np.random.default_rng(0), 50)
        assert np.isfinite(m) and m > .5


# ------------------------------------------------------------------ ANOVA
class TestAnova:
    def setup_method(self):
        """Image x 2 x 2 data with an effect of B and an interaction."""
        rng = np.random.default_rng(6)
        self.Y = rng.normal(size=(50, 2, 2))
        self.Y[:, :, 1] += .4                  # effect of B
        self.Y[:, 1, 1] += .3                  # interaction

    def test_F_equals_t_squared(self):
        """In a 2 x 2 within design every F is t^2 of the matching contrast."""
        Y = self.Y
        res = S.anova_2x2(Y)
        contrasts = {"A": Y[:, 0, :].mean(1) - Y[:, 1, :].mean(1),
                     "B": Y[:, :, 0].mean(1) - Y[:, :, 1].mean(1),
                     "AxB": (Y[:, 0, 0] - Y[:, 0, 1]) - (Y[:, 1, 0] - Y[:, 1, 1])}
        for k, c in contrasts.items():
            t = stats.ttest_1samp(c, 0)
            F, df1, df2, p, eta = res[k]
            assert F == pytest.approx(t.statistic ** 2)
            assert p == pytest.approx(t.pvalue)
            assert (df1, df2) == (1, 49)
            assert eta == pytest.approx(F / (F + df2))

    def test_tiny_p_is_not_zero(self):
        """An enormous effect gives a p that is tiny but above zero."""
        Y = self.Y.copy()
        Y[:, :, 1] += 50                       # enormous effect
        p = S.anova_2x2(Y)["B"][3]
        assert 0 < p < 1e-20


class TestEta2pCI:
    def test_lower_bound_zero_iff_not_significant(self):
        """The 90% lower bound exceeds 0 iff p < .05, and the CI brackets the
        point estimate."""
        for F in (0.5, 2.0, 3.0, 3.9, 4.1, 6.0, 15.0):
            lo, hi = S.eta2p_ci(F, 1, 100)
            p = stats.f.sf(F, 1, 100)
            assert (lo > 0) == (p < .05), F
            assert 0 <= lo < F / (F + 100) < hi < 1

    def test_small_F(self):
        """A near-zero F gives a CI from 0 to a positive upper bound."""
        lo, hi = S.eta2p_ci(0.01, 1, 50)
        assert lo == 0 and hi > 0


def test_spearman_ci():
    """The Spearman CI is symmetric in Fisher z and wider than Pearson's."""
    lo, hi = S.spearman_ci(.5, 100)
    assert lo < .5 < hi
    # symmetric on the Fisher z scale
    assert np.arctanh(.5) - np.arctanh(lo) == pytest.approx(np.arctanh(hi) - np.arctanh(.5))
    # wider than the Pearson interval
    se_p = 1 / np.sqrt(97)
    assert hi - lo > np.tanh(np.arctanh(.5) + 1.96 * se_p) - np.tanh(np.arctanh(.5) - 1.96 * se_p)


def test_per_image_squared_error():
    """Per-image error is the mean held-out squared error over its folds."""
    rng = np.random.default_rng(7)
    n = 40
    z = rng.normal(size=(n, 1))
    y = 2 * z[:, 0] + rng.normal(size=n)
    splits = E.make_splits(n, rng, n_repeats=5)
    got = S.per_image_squared_error(z, y, splits, n)
    total, count = np.zeros(n), np.zeros(n)
    for train in splits:
        test = np.setdiff1d(np.arange(n), train)
        pred = E.ols_predict(z[train], y[train], z[test])
        total[test] += (y[test] - pred) ** 2
        count[test] += 1
    assert np.allclose(got, total / count)
    assert np.all(count == 5)


# ----------------------------------------------------------------------- nan
def test_nan_reductions_are_quiet():
    """nan.mean and nan.std give NaN for all-NaN columns without warnings."""
    a = np.array([[np.nan, 1.0], [np.nan, 3.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = nan.mean(a)
        s = nan.std(a)
    assert np.isnan(m[0]) and m[1] == 2
    assert np.isnan(s[0]) and s[1] == pytest.approx(np.std([1, 3], ddof=1))
