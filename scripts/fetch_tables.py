"""Download the tables (53 files, 40 GB) from their Zenodo record into tables/redshift/ and check every file against
data/tables/tables.sha256.  A file already present with the right hash is skipped, so the download can be resumed.

    python scripts/fetch_tables.py <zenodo record id>
"""
import hashlib
import json
import os
import sys
import urllib.request

record = sys.argv[1]
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target = os.path.join(root, "tables", "redshift")
os.makedirs(target, exist_ok=True)
manifest = dict(line.split()[::-1] for line in open(os.path.join(root, "data", "tables", "tables.sha256")) if ".npz" in line)
with urllib.request.urlopen("https://zenodo.org/api/records/%s" % record, timeout=60) as r:
    links = {f["key"]: f["links"]["self"] for f in json.load(r)["files"]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


for name, expected in sorted(manifest.items()):
    path = os.path.join(target, name)
    if os.path.exists(path) and sha256(path) == expected:
        continue
    print("downloading", name, flush=True)
    urllib.request.urlretrieve(links[name], path + ".part")
    if sha256(path + ".part") != expected:
        raise RuntimeError("wrong hash for " + name)
    os.replace(path + ".part", path)
print("tables complete: %d files in %s" % (len(manifest), target))
