#!/usr/bin/env python3
"""
Download the released dataset from Zenodo and unpack it into data/.

The archive and its checksum are pinned in config.ZENODO_URL and
config.ZENODO_HASH. An existing data/ folder is left unchanged.

Run: python3 fetch_data.py
"""

from facescenes import download


def main():
    print(f"dataset at {download.fetch()}")


if __name__ == "__main__":
    main()
