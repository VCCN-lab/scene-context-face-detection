"""Unit tests for the encoding models, on synthetic data."""

import numpy as np
import pytest
from scipy import stats

from facescenes import config as cfg
from facescenes import encoding as E


class TestSignedR2:
    def test_perfect_and_reversed(self):
        y = np.arange(10.0)
        assert E.signed_r2(y, y ** 3) == pytest.approx(1)       # rank-based
        assert E.signed_r2(y, -y) == pytest.approx(-1)

    def test_matches_scipy_spearman(self):
        rng = np.random.default_rng(0)
        y, yhat = rng.normal(size=50), rng.normal(size=50)
        r = stats.spearmanr(y, yhat).statistic
        assert E.signed_r2(y, yhat) == pytest.approx(np.sign(r) * r * r)

    def test_ties(self):
        y = np.array([1, 2, 2, 3, 4.0])
        yhat = np.array([1, 1, 2, 3, 3.0])
        r = stats.spearmanr(y, yhat).statistic
        assert E.signed_r2(y, yhat) == pytest.approx(np.sign(r) * r * r)

    def test_constant_prediction_is_zero(self):
        assert E.signed_r2(np.arange(5.0), np.ones(5)) == 0.0


def test_ols_predict_recovers_line():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(30, 2))
    y = 1.5 + x @ [2.0, -3.0]
    xt = rng.normal(size=(5, 2))
    assert np.allclose(E.ols_predict(x, y, xt), 1.5 + xt @ [2.0, -3.0])


class TestMakeSplits:
    @pytest.mark.parametrize("n", [108, 120, 7])
    def test_balanced(self, n):
        splits = E.make_splits(n, np.random.default_rng(2), n_repeats=50)
        assert len(splits) == 100
        held_out = np.zeros(n, int)
        for r in range(50):
            a, b = splits[2 * r], splits[2 * r + 1]
            # the two folds of a repetition are complementary halves
            assert len(np.intersect1d(a, b)) == 0      # disjoint training sets
            assert sorted(np.union1d(a, b)) == list(range(n))
            assert abs(len(a) - len(b)) <= 1
            for train in (a, b):
                held_out[np.setdiff1d(np.arange(n), train)] += 1
        assert np.all(held_out == 50)

    def test_seeded(self):
        a = E.make_splits(20, np.random.default_rng(5), 3)
        b = E.make_splits(20, np.random.default_rng(5), 3)
        assert all(np.array_equal(x, y) for x, y in zip(a, b))


class TestFoldScores:
    def setup_method(self):
        rng = np.random.default_rng(3)
        self.n, self.P = 80, 4
        self.Z = rng.normal(size=(self.n, self.P))
        # feature 0 is strongly predictive, the rest are noise
        self.y = 2 * self.Z[:, 0] + .5 * rng.normal(size=self.n)
        self.splits = E.make_splits(self.n, rng, n_repeats=10)

    def test_shapes_and_excluded_column(self):
        s, u = E.fold_scores(self.Z, self.y, self.splits, [0, 1, 2])
        assert s.shape == u.shape == (20, 4)
        assert np.all(np.isnan(u[:, 3]))       # not in the full model
        assert not np.any(np.isnan(u[:, :3]))
        assert not np.any(np.isnan(s))

    def test_single_matches_manual(self):
        s, _ = E.fold_scores(self.Z, self.y, self.splits, [0, 1, 2])
        train = self.splits[0]
        test = np.setdiff1d(np.arange(self.n), train)
        k = 2
        pred = E.ols_predict(self.Z[train][:, [k]], self.y[train], self.Z[test][:, [k]])
        assert s[0, k] == pytest.approx(E.signed_r2(self.y[test], pred))

    def test_signal_is_found(self):
        s, u = E.fold_scores(self.Z, self.y, self.splits, [0, 1, 2, 3])
        assert s[:, 0].mean() > .7
        assert u[:, 0].mean() > .5
        assert np.all(np.abs(s[:, 1:].mean(0)) < .1)
        assert np.all(np.abs(u[:, 1:].mean(0)) < .05)

    def test_invariant_to_monotone_transform(self):
        """Single-feature predictivity depends only on the sign of the fit and
        the ranks, so a monotone transform of a lone predictor changes nothing
        as long as the fitted slope keeps its sign."""
        Z2 = self.Z.copy()
        Z2[:, 0] = np.exp(Z2[:, 0])
        s1, _ = E.fold_scores(self.Z, self.y, self.splits, [0])
        s2, _ = E.fold_scores(Z2, self.y, self.splits, [0])
        assert np.allclose(s1[:, 0], s2[:, 0])


def test_standardised_betas():
    rng = np.random.default_rng(4)
    n = 100
    Z = rng.normal(size=(n, 3))
    y = Z[:, 0] - .5 * Z[:, 1] + rng.normal(size=n)
    splits = E.make_splits(n, rng, n_repeats=3)
    single, full = E.standardised_betas(Z, y, splits, [0, 1])
    # with x and y both z-scored, the single-feature slope is Pearson's r
    tr = splits[0]
    assert single[0, 0] == pytest.approx(np.corrcoef(Z[tr, 0], y[tr])[0, 1])
    assert np.all(np.isnan(full[:, 2]))
    assert np.all(full[:, 0] > 0) and np.all(full[:, 1] < 0)


def test_noise_correct():
    rel = cfg.RELIABILITY["facedetection"]["preview"]
    assert E.noise_correct(np.array([rel]), "facedetection", "preview")[0] == pytest.approx(1)


def test_unique_columns():
    cols = E.unique_columns()
    assert len(cols) == len(cfg.FEATURE_NAMES) - 1
    assert cfg.FEATURE_NAMES.index("UntrainedCNN") not in cols


def test_bootstrap_splits_keep_copies_together():
    import run_statistics_tables as T
    rng = np.random.default_rng(6)
    idx = rng.integers(0, 30, 30)                  # resample with duplicates
    for train in T.bootstrap_splits(idx, rng, n_repeats=5):
        test = np.setdiff1d(np.arange(len(idx)), train)
        # no image may sit in both halves through two of its copies
        assert not set(idx[train]) & set(idx[test])
