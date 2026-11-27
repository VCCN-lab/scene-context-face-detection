"""Tests on where the code reads the released dataset and writes its results.

The released dataset (data/, with its derivatives in data/derivatives/) is read
only. Scripts write their results to cfg.OUTPUT, and readers of cached results
take a fresh result from cfg.OUTPUT when one exists and the released one
otherwise.
"""

import hashlib

import numpy as np
import pytest

from facescenes import config as cfg
from facescenes import data as D
import run_contrast_test as CT
import run_encoding_analyses as ENC

TASK = "facedetection"


def digest(path):
    """The md5 hex digest of the file at `path`."""
    return hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def output(tmp_path, monkeypatch):
    """Point cfg.OUTPUT at a fresh folder under `tmp_path` and return it."""
    out = tmp_path / "derivatives"
    monkeypatch.setattr(cfg, "OUTPUT", out)
    return out


# ------------------------------------------------------------ layout
def test_data_root_defaults_to_repository_data():
    """Without FACESCENES_DATA the data root is the repository's data/."""
    assert cfg.data_root({}) == cfg.ROOT / "data"


def test_data_root_follows_environment(tmp_path):
    """FACESCENES_DATA sets the data root."""
    env = {"FACESCENES_DATA": str(tmp_path / "elsewhere")}
    assert cfg.data_root(env) == tmp_path / "elsewhere"


def test_derivatives_are_nested_in_the_dataset():
    """Derivatives and stimulus features live inside the dataset folder."""
    assert cfg.DERIVATIVES == cfg.DATA / "derivatives"
    assert cfg.FEATURES == cfg.DERIVATIVES / "stimulus_features"


def test_output_is_outside_the_dataset():
    """Results are written to derivatives/ outside the dataset."""
    assert cfg.OUTPUT == cfg.ROOT / "derivatives"
    assert cfg.DATA not in cfg.OUTPUT.parents and cfg.OUTPUT != cfg.DATA


# ---------------------------------------------------- cached results
def test_results_path_falls_back_to_release(output):
    """Without a fresh result the released one is read."""
    name = f"{TASK}_encoding.npz"
    released = cfg.DERIVATIVES / "encoding" / name
    assert D.results_path("encoding", name) == released


def test_results_path_prefers_fresh_result(output):
    """A fresh result in cfg.OUTPUT is read before the released one."""
    name = f"{TASK}_encoding.npz"
    fresh = output / "encoding" / name
    fresh.parent.mkdir(parents=True)
    fresh.write_bytes(b"")
    assert D.results_path("encoding", name) == fresh


def test_analyse_reads_release_without_writing(output):
    """analyse returns the released results and writes nothing."""
    released = cfg.DERIVATIVES / "encoding" / f"{TASK}_encoding.npz"
    res = ENC.analyse(TASK, n_perm=2)
    expected = np.load(released, allow_pickle=False)
    assert set(res) == set(expected.files)
    assert np.allclose(res["single_preview"], expected["single_preview"])
    assert not output.exists()


def test_analyse_force_writes_output_only(output):
    """analyse with force writes to cfg.OUTPUT and leaves the release
    unchanged."""
    released = cfg.DERIVATIVES / "encoding" / f"{TASK}_encoding.npz"
    before = digest(released)
    ENC.analyse(TASK, n_perm=2, force=True)
    assert (output / "encoding" / released.name).is_file()
    assert digest(released) == before


def test_contrast_test_writes_output_only(output):
    """The contrast test writes to cfg.OUTPUT and leaves the release
    unchanged."""
    released = cfg.DERIVATIVES / "encoding" / f"{TASK}_contrast.npz"
    before = digest(released)
    CT.run(TASK, n_perm=2)
    assert (output / "encoding" / released.name).is_file()
    assert digest(released) == before


# ------------------------------------------------------------ loading
def test_no_pickle_loading():
    """Arrays are loaded with numpy's default allow_pickle=False."""
    sources = sorted((cfg.ROOT / "code").rglob("*.py"))
    sources += sorted((cfg.ROOT / "tests").glob("*.py"))
    offenders = [p.name for p in sources
                 if "allow_pickle=" + "True" in p.read_text(encoding="utf-8")]
    assert offenders == []
