"""Tests on the metadata of the released dataset and its derivatives.

The dataset root (cfg.DATA) has a README and a .bidsignore whose patterns all
match released files. The event tables are behavioural data (beh/), name the
scenes in a plain `stimulus` column (the photographs are not released), and
their sidecars document every column, the task and the stimulus
presentation. Every Levels entry covers the values used. Every folder in its
derivatives/ is a BIDS derivative dataset with its own .bidsignore and its own
dataset_description.json. Every data file in it has a JSON sidecar with the
same name. Tables are BIDS TSV files: tab-separated, with "n/a" for missing
values. A table's sidecar documents each column, in order, BIDS style. An
array file's sidecar has a file-level "Description" and an "Arrays" entry
documenting each array with its "Shape"; a .npy file holds one array, named
after the file.
"""

import fnmatch
import json
import re

import numpy as np
import pandas as pd
import pytest
import scipy.io as sio

from facescenes import config as cfg

DERIVATIVES = cfg.DERIVATIVES
PIPELINES = ["encoding", "source_data", "stimulus_features", "tables"]
ARRAY_SUFFIXES = {".npz", ".npy", ".mat"}
DATA_SUFFIXES = ARRAY_SUFFIXES | {".tsv"}
DATA_FILES = sorted(p for name in PIPELINES
                    for p in (DERIVATIVES / name).iterdir()
                    if p.suffix in DATA_SUFFIXES)
TABLES = [p for p in DATA_FILES if p.suffix == ".tsv"]
ARRAYS = [p for p in DATA_FILES if p.suffix in ARRAY_SUFFIXES]
EVENTS = sorted(cfg.DATA.glob("sub-*/*/sub-*_events.tsv"))
OSF_STIMULI = "10.17605/OSF.IO/5AYX8"
TASK_KEYS = {"StimulusPresentation", "TaskName", "TaskDescription"}


def file_id(path):
    return f"{path.parent.name}/{path.name}"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_table(path):
    return pd.read_csv(path, sep="\t")


def ignored(path, pattern):
    """gitignore-style match: patterns without a slash match the name."""
    pattern = pattern.strip("/")
    target = path.as_posix() if "/" in pattern else path.name
    return fnmatch.fnmatch(target, pattern)


def bidsignore_patterns(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    return [s.strip() for s in lines if s.strip() and not s.startswith("#")]


def levels_violations(table, meta):
    """Values of Levels columns that their Levels do not list."""
    out = {}
    for column, entry in meta.items():
        if column in table and "Levels" in entry:
            used = set(table[column].dropna().astype(str))
            extra = sorted(used - set(entry["Levels"]))
            if extra:
                out[column] = extra
    return out


def event_columns(task):
    meta = read_json(cfg.DATA / f"task-{task}_events.json")
    return [k for k in meta if k not in TASK_KEYS]


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
    return read_json(cfg.DATA / "dataset_description.json")


# ---------------------------------------------------------- dataset root
def test_readme(raw_description):
    text = (cfg.DATA / "README.md").read_text(encoding="utf-8")
    assert raw_description["License"] in text
    assert "https://github.com/VCCN-lab/scene-context-face-detection" in text
    assert "derivatives/" in text


def test_bidsignore_patterns_match_released_files():
    patterns = bidsignore_patterns(cfg.DATA / ".bidsignore")
    assert patterns
    files = [p.relative_to(cfg.DATA) for p in cfg.DATA.rglob("*")]
    for pattern in patterns:
        assert any(ignored(f, pattern) for f in files), pattern


def test_raw_source_is_osf_stimuli(raw_description):
    osf = {"DOI": f"doi:{OSF_STIMULI}",
           "URL": f"https://doi.org/{OSF_STIMULI}"}
    assert osf in raw_description["SourceDatasets"]


def test_participant_levels():
    table = pd.read_csv(cfg.DATA / "participants.tsv", sep="\t", dtype=str)
    meta = read_json(cfg.DATA / "participants.json")
    assert levels_violations(table, meta) == {}


# ------------------------------------------------------------- events
def test_events_are_behavioural_data():
    assert len(EVENTS) == 76
    assert {p.parent.name for p in EVENTS} == {"beh"}
    assert not list(cfg.DATA.glob("sub-*/eyetrack"))


@pytest.mark.parametrize("task", cfg.TASKS)
def test_events_sidecar_documents_every_column(task):
    columns = event_columns(task)
    for path in EVENTS:
        if f"_task-{task}_" in path.name:
            header = path.open(encoding="utf-8").readline()
            assert header.rstrip("\n").split("\t") == columns, path.name


def test_events_name_stimuli_without_file_reference():
    order = set(read_table(cfg.DATA / "stimulus_order.tsv")["stimulus"])
    for path in EVENTS:
        events = read_table(path)
        assert "stim_file" not in events.columns, path.name
        assert set(events["stimulus"]) <= order, path.name


@pytest.mark.parametrize("task", cfg.TASKS)
def test_events_stimulus_column_points_to_osf(task):
    meta = read_json(cfg.DATA / f"task-{task}_events.json")
    assert OSF_STIMULI in meta["stimulus"]["Description"]


@pytest.mark.parametrize("task", cfg.TASKS)
def test_event_levels(task):
    meta = read_json(cfg.DATA / f"task-{task}_events.json")
    for path in EVENTS:
        if f"_task-{task}_" in path.name:
            table = pd.read_csv(path, sep="\t", dtype=str)
            assert levels_violations(table, meta) == {}, path.name


@pytest.mark.parametrize("task", cfg.TASKS)
def test_task_name_gives_the_task_label(task):
    """BIDS: the label is TaskName without its non-alphanumeric characters."""
    meta = read_json(cfg.DATA / f"task-{task}_events.json")
    assert re.sub(r"[^A-Za-z0-9]", "", meta["TaskName"]) == task


@pytest.mark.parametrize("task", cfg.TASKS)
def test_task_description_states_the_time_limit(task):
    meta = read_json(cfg.DATA / f"task-{task}_events.json")
    assert f"{cfg.STIM_DUR[task]:g} s" in meta["TaskDescription"]


@pytest.mark.parametrize("task", cfg.TASKS)
def test_stimulus_presentation(task):
    sp = read_json(cfg.DATA / f"task-{task}_events.json")
    sp = sp["StimulusPresentation"]
    d = cfg.DISPLAY
    assert sp["ScreenDistance"] == pytest.approx(
        d["viewing_distance_mm"] / 1000)
    assert sp["ScreenSize"] == pytest.approx(
        [d["width_mm"] / 1000, d["height_mm"] / 1000])
    assert sp["ScreenResolution"] == [d["width_px"], d["height_px"]]
    assert sp["ScreenRefreshRate"] == d["refresh_hz"]
    assert sp["SoftwareName"] == "Psychtoolbox"


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
             and p.suffix != ".json" and p.name != ".bidsignore"]
    assert other == []
    assert sidecars - {"dataset_description"} == data


@pytest.mark.parametrize("pipeline", PIPELINES)
def test_derivative_bidsignore(pipeline):
    """The derivative's own .bidsignore covers every file except its
    dataset_description.json, and every pattern matches a file."""
    folder = DERIVATIVES / pipeline
    patterns = bidsignore_patterns(folder / ".bidsignore")
    files = [p.relative_to(folder) for p in folder.iterdir()
             if p.is_file() and p.name != ".bidsignore"]
    for pattern in patterns:
        assert any(ignored(f, pattern) for f in files), pattern
    for f in files:
        covered = any(ignored(f, pattern) for pattern in patterns)
        assert covered == (f.name != "dataset_description.json"), f.name


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
