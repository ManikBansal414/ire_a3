"""Loading the ann-benchmarks HDF5 files and splitting the queries."""
import h5py
import numpy as np

from . import config

FILES = {
    "sift": "sift-128-euclidean.hdf5",
    "glove": "glove-100-angular.hdf5",
}


def normalise(x):
    """L2-normalise rows. Zero rows (there are none in GloVe-100, but be safe) stay zero."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(x / norms, dtype=np.float32)


def load(name, n_base=None):
    """Return dict with base, queries, stored neighbours/distances and metadata.

    GloVe is normalised so that Euclidean distance ranks exactly like angular
    distance (||a-b||^2 = 2 - 2 cos(a,b) for unit vectors), and Euclidean
    distance is used for everything.
    """
    path = config.DATA_DIR / FILES[name]
    with h5py.File(path, "r") as f:
        base = f["train"][: n_base] if n_base else f["train"][:]
        queries = f["test"][:]
        stored_nn = f["neighbors"][:]
        stored_d = f["distances"][:]
        metric = f.attrs["distance"]
    base = np.ascontiguousarray(base, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if name == "glove":
        base, queries = normalise(base), normalise(queries)
    return {
        "name": name,
        "base": base,
        "queries": queries,
        "stored_nn": stored_nn,        # only valid when the full base set is used
        "stored_d": stored_d,
        "metric_in_file": metric,
        "subset": n_base is not None,
    }


def split_queries(q, n_tune=config.N_QUERIES_SPLIT):
    """First half for tuning, second half for the reported numbers."""
    return q[:n_tune], q[n_tune:]
