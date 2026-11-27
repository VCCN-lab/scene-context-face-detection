"""Tests on downloading and unpacking the released dataset.

A small archive with the layout of the release is served from a local HTTP
server, so these tests need no network access. The test marked `zenodo`
downloads the published record and is run only on request:
`uv run pytest -m zenodo`.
"""

import functools
import hashlib
import http.server
import threading
import zipfile

import pytest

from facescenes import config as cfg
from facescenes import download as DL

FILES = {"data/dataset_description.json": '{"Name": "test"}\n',
         "data/derivatives/encoding/dataset_description.json": "{}\n"}


def md5(path):
    """The md5 checksum of `path`, prefixed with "md5:"."""
    return "md5:" + hashlib.md5(path.read_bytes()).hexdigest()


@pytest.fixture
def archive(tmp_path):
    """A zip archive of FILES in its own folder under `tmp_path`."""
    folder = tmp_path / "served"
    folder.mkdir()
    path = folder / "data.zip"
    with zipfile.ZipFile(path, "w") as z:
        for name, text in FILES.items():
            z.writestr(name, text)
    return path


@pytest.fixture
def url(archive):
    """URL of `archive` served by a local HTTP server for the test."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler,
                                directory=str(archive.parent))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/{archive.name}"
    server.shutdown()
    server.server_close()


def test_fetch_unpacks_the_dataset(tmp_path, archive, url):
    """fetch unpacks the archive and returns the data folder."""
    dest = tmp_path / "repo"
    data = DL.fetch(url, md5(archive), dest=dest, cache=tmp_path / "cache")
    assert data == dest / "data"
    for name, text in FILES.items():
        assert (dest / name).read_text() == text


def test_fetch_keeps_the_archive_in_the_cache(tmp_path, archive, url):
    """fetch leaves the downloaded zip archive in the cache."""
    cache = tmp_path / "cache"
    DL.fetch(url, md5(archive), dest=tmp_path / "repo", cache=cache)
    assert [p.name for p in cache.iterdir() if p.suffix == ".zip"]


def test_fetch_rejects_a_wrong_checksum(tmp_path, url):
    """A checksum mismatch raises ValueError and unpacks nothing."""
    dest = tmp_path / "repo"
    with pytest.raises(ValueError):
        DL.fetch(url, "md5:" + "0" * 32, dest=dest, cache=tmp_path / "cache")
    assert not (dest / "data").exists()


def test_fetch_fills_an_empty_data_folder(tmp_path, archive, url):
    """An empty data/, e.g. a mount point, does not count as a dataset."""
    dest = tmp_path / "repo"
    (dest / "data").mkdir(parents=True)
    data = DL.fetch(url, md5(archive), dest=dest, cache=tmp_path / "cache")
    assert data == dest / "data"
    for name, text in FILES.items():
        assert (dest / name).read_text() == text


def test_fetch_leaves_existing_data_untouched(tmp_path, archive, url):
    """A data folder that holds files is kept unchanged, with a warning."""
    dest = tmp_path / "repo"
    (dest / "data").mkdir(parents=True)
    (dest / "data" / "mine.txt").write_text("keep")
    with pytest.warns(UserWarning):
        data = DL.fetch(url, md5(archive), dest=dest,
                        cache=tmp_path / "cache")
    assert data == dest / "data"
    assert sorted(p.name for p in data.iterdir()) == ["mine.txt"]


def test_fetch_needs_a_published_record(tmp_path, monkeypatch):
    """Without a configured Zenodo URL fetch raises RuntimeError."""
    monkeypatch.setattr(cfg, "ZENODO_URL", None)
    with pytest.raises(RuntimeError):
        DL.fetch(dest=tmp_path / "repo", cache=tmp_path / "cache")


@pytest.mark.zenodo
def test_fetch_published_record(tmp_path):
    """The published Zenodo record unpacks into a BIDS dataset."""
    data = DL.fetch(dest=tmp_path / "repo", cache=tmp_path / "cache")
    assert (data / "dataset_description.json").is_file()
    assert (data / "derivatives" / "encoding").is_dir()
