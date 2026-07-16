"""Portable, inexpensive identity checks for aligned residual datasets."""

from __future__ import annotations

import hashlib
from pathlib import Path

import h5py
import numpy as np


def file_sha256(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash the complete file so a relocated byte-identical HDF5 is accepted."""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as stream:
        while True:
            chunk = stream.read(int(chunk_bytes))
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def action_dataset_sha256(path: str | Path, action_key: str = "actions") -> str:
    """Hash demo names, action shapes/dtypes, and action bytes.

    Actions are small compared with the image observations, so this gives a
    portable content identity without hashing multi-gigabyte camera data.
    """

    digest = hashlib.sha256()
    path = Path(path).expanduser().resolve()
    with h5py.File(path, "r") as source:
        if "data" not in source:
            raise KeyError(f"HDF5 file has no data group: {path}")
        names = sorted(
            source["data"],
            key=lambda name: (
                0,
                int(name.rsplit("_", 1)[-1]),
            )
            if name.rsplit("_", 1)[-1].isdigit()
            else (1, name),
        )
        for name in names:
            if action_key not in source["data"][name]:
                raise KeyError(f"Missing {action_key!r} in {name}: {path}")
            dataset = source["data"][name][action_key]
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(dataset.shape)).encode("ascii"))
            digest.update(str(dataset.dtype).encode("ascii"))
            array = np.asarray(dataset)
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


__all__ = ["action_dataset_sha256", "file_sha256"]
