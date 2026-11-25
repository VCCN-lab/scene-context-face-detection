"""Tests on the metadata of the derivative datasets.

Every folder in derivatives/ is a BIDS derivative dataset with its own
dataset_description.json. Every data file in it has a JSON sidecar with the
same name. Tables are BIDS TSV files: tab-separated, with "n/a" for missing
values. A table's sidecar documents each column, in order, BIDS style. An
array file's sidecar has a file-level "Description" and an "Arrays" entry
documenting each array with its "Shape"; a .npy file holds one array, named
after the file.
"""

import json

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from facescenes import config as cfg

DERIVATIVES = cfg.ROOT / "derivatives"
PIPELINES = ["encoding", "source_data", "stimulus_features", "tables"]
ARRAY_SUFFIXES = {".npz", ".npy", ".mat"}
DATA_SUFFIXES = ARRAY_SUFFIXES | {".tsv"}
DATA_FILES = sorted(p for name in PIPELINES
                    for p in (DERIVATIVES / name).iterdir()
                    if p.suffix in DATA_SUFFIXES)
TABLES = [p for p in DATA_FILES if p.suffix == ".tsv"]
ARRAYS = [p for p in DATA_FILES if p.suffix in ARRAY_SUFFIXES]


def file_id(path):
    return f"{path.parent.name}/{path.name}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path):
    return pd.read_csv(path, sep="\t")


def sidecar(path):
    return read_json(path.with_suffix(".json"))


def array_shapes(path):
    if path.suffix == ".npz":
        arrays = np.load(path, allow_pickle=False)
        return {k: arrays[k].shape for k in arrays.files}
    if path.suffix == ".npy":
        return {path.stem: np.load(path, allow_pickle=False).shape}
    return {name: shape for name, shape, _ in sio.whosmat(path)}


@pytest.fixture(scope="module")
def raw_description():
    return read_json(cfg.ROOT / "data" / "dataset_description.json")


# ------------------------------------------------------- dataset level
def test_derivative_folders():
    found = sorted(p.name for p in DERIVATIVES.iterdir() if p.is_dir())
    assert found == PIPELINES


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_dataset_description(pipeline, raw_description):
    desc = read_json(DERIVATIVES / pipeline / "dataset_description.json")
    assert desc["Name"].strip()
    assert desc["DatasetType"] == "derivative"
    assert desc["BIDSVersion"] == raw_description["BIDSVersion"]
    assert desc["License"] == raw_description["License"]
    assert desc["Authors"]
    assert desc["GeneratedBy"]
    assert all(g["Name"].strip() for g in desc["GeneratedBy"])
    assert desc["SourceDatasets"]
    assert all("URL" in s or "DOI" in s for s in desc["SourceDatasets"])


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_generating_code_exists(pipeline):
    """Code named in GeneratedBy is in code/ or notebooks/."""
    desc = read_json(DERIVATIVES / pipeline / "dataset_description.json")
    for name in (g["Name"] for g in desc["GeneratedBy"]):
        if name.endswith((".py", ".ipynb")):
            assert any((cfg.ROOT / d / name).is_file()
                       for d in ("code", "notebooks")), name


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_every_file_is_documented(pipeline):
    """Only known file types, each with a sidecar, and no orphan sidecars."""
    files = [p for p in (DERIVATIVES / pipeline).iterdir() if p.is_file()]
    data = {p.stem for p in files if p.suffix in DATA_SUFFIXES}
    sidecars = {p.stem for p in files if p.suffix == ".json"}
    other = [p.name for p in files if p.suffix not in DATA_SUFFIXES
             and p.suffix != ".json"]
    assert other == []
    assert sidecars - {"dataset_description"} == data


# ----------------------------------------------------------- tables
@pytest.mark.parametrize("path", TABLES, ids=file_id)
def test_table_sidecar_columns(path):
    columns = read_table(path).columns
    meta = sidecar(path)
    assert list(meta) == list(columns)
    assert all(entry["Description"].strip() for entry in meta.values())


@pytest.mark.parametrize("path", TABLES, ids=file_id)
def test_table_sidecar_levels(path):
    """Repeated text values carry Levels, and Levels cover every value."""
    table = read_table(path)
    meta = sidecar(path)
    for column in table.columns:
        values = table[column].dropna()
        if "Levels" in meta[column]:
            assert set(values.astype(str)) <= set(meta[column]["Levels"])
        elif pd.api.types.is_string_dtype(values):
            assert values.is_unique, column


@pytest.mark.parametrize("path", TABLES, ids=file_id)
def test_table_is_bids_tsv(path):
    """Every row has one field per column and no field is empty."""
    lines = path.read_text(encoding="utf-8").splitlines()
    n_columns = len(lines[0].split("\t"))
    for line in lines:
        fields = line.split("\t")
        assert len(fields) == n_columns
        assert all(f.strip() for f in fields)


# ----------------------------------------------------------- arrays
@pytest.mark.parametrize("path", ARRAYS, ids=file_id)
def test_array_sidecar(path):
    meta = sidecar(path)
    assert meta["Description"].strip()
    documented = {name: tuple(entry["Shape"])
                  for name, entry in meta["Arrays"].items()}
    assert documented == array_shapes(path)
    assert all(e["Description"].strip() for e in meta["Arrays"].values())
