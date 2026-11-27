"""Tests on how run_statistics_tables.py writes the Supplementary Tables."""

import numpy as np
import pandas as pd

from facescenes import config as cfg
import run_statistics_tables as ST


def test_write_table_writes_bids_tsv(tmp_path, monkeypatch):
    """write_table writes a tab-separated .tsv with "n/a" for NaN."""
    monkeypatch.setattr(cfg, "OUTPUT", tmp_path)
    table = pd.DataFrame(dict(task=["facedetection"], p=[np.nan],
                              r2=[0.12345678912345]))
    path = ST.write_table(table, "example")
    assert path == tmp_path / "tables" / "example.tsv"
    assert path.read_text().splitlines() == [
        "task\tp\tr2", "facedetection\tn/a\t0.12345678912345"]
    assert not (tmp_path / "tables" / "example.csv").exists()


def test_write_table_round_trips(tmp_path, monkeypatch):
    """A written table reads back unchanged."""
    monkeypatch.setattr(cfg, "OUTPUT", tmp_path)
    table = pd.DataFrame(dict(feature=["Face expect.", "Pixel"],
                              p=[0.0002, np.nan], df1=[1, 1]))
    back = pd.read_csv(ST.write_table(table, "example"), sep="\t")
    pd.testing.assert_frame_equal(back, table)
