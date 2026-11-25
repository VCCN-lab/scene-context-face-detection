"""Tests on how run_statistics_tables.py writes the Supplementary Tables."""

import numpy as np
import pandas as pd

import run_statistics_tables as ST


def test_write_table_writes_bids_tsv(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "OUT", tmp_path)
    table = pd.DataFrame(dict(task=["facedetection"], p=[np.nan],
                              r2=[0.12345678912345]))
    path = ST.write_table(table, "example")
    assert path == tmp_path / "example.tsv"
    assert path.read_text().splitlines() == [
        "task\tp\tr2", "facedetection\tn/a\t0.12345678912345"]
    assert not (tmp_path / "example.csv").exists()


def test_write_table_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(ST, "OUT", tmp_path)
    table = pd.DataFrame(dict(feature=["Face expect.", "Pixel"],
                              p=[0.0002, np.nan], df1=[1, 1]))
    back = pd.read_csv(ST.write_table(table, "example"), sep="\t")
    pd.testing.assert_frame_equal(back, table)
