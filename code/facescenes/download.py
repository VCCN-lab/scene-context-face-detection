"""
Downloading the released dataset.

The dataset is published on Zenodo as one zip archive whose only top-level
folder is data/, the BIDS dataset with its derivatives in data/derivatives/.
fetch() downloads the archive into a cache, verifies its checksum, and unpacks
it next to the code, so that cfg.DATA finds it.
"""

import warnings
import zipfile
from pathlib import Path

import pooch

from . import config as cfg


def fetch(url=None, known_hash=None, dest=None, cache=None):
    """Download the release archive, unpack it into `dest`, return dest/data.

    `url` and `known_hash` default to cfg.ZENODO_URL and cfg.ZENODO_HASH,
    `dest` to the repository root and `cache` to the user cache folder. The
    archive is kept in `cache`, so a second call does not download it again.
    A dest/data that holds files is left unchanged; an empty one, such as a
    mount point, is filled.
    """
    url = cfg.ZENODO_URL if url is None else url
    known_hash = cfg.ZENODO_HASH if known_hash is None else known_hash
    dest = cfg.ROOT if dest is None else Path(dest)
    if url is None:
        raise RuntimeError("no published Zenodo record is configured "
                           "(config.ZENODO_URL); set FACESCENES_DATA to a "
                           "local copy of the dataset instead")
    data = dest / "data"
    if data.exists() and any(data.iterdir()):
        warnings.warn(f"{data} already holds files and is left unchanged; "
                      "remove it to unpack the release again")
        return data
    cache = pooch.os_cache("facescenes") if cache is None else Path(cache)
    archive = pooch.retrieve(url, known_hash, path=cache, progressbar=False)
    unpack(archive, dest)
    return data


def unpack(archive, dest):
    """Extract `archive` into `dest`; every member must lie inside data/."""
    with zipfile.ZipFile(archive) as z:
        outside = [n for n in z.namelist() if not n.startswith("data/")]
        if outside:
            raise ValueError(f"{archive} has members outside data/: "
                             f"{outside[:5]}")
        z.extractall(dest)
