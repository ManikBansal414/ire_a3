"""Download the two ann-benchmarks HDF5 files into the data directory if missing."""
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from a3 import config  # noqa: E402
from a3.data import FILES  # noqa: E402

URL = "http://ann-benchmarks.com/{}"


def main():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES.values():
        path = config.DATA_DIR / fname
        if path.exists():
            print(f"{path} present")
            continue
        print(f"downloading {fname} ...", flush=True)
        urllib.request.urlretrieve(URL.format(fname), path.with_suffix(".part"))
        os.replace(path.with_suffix(".part"), path)


if __name__ == "__main__":
    main()
