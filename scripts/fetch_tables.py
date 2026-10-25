"""Download files of the Zenodo record and check each one against its SHA-256 list.  A file already present with the
right hash is skipped, so a download can be resumed.

    python scripts/fetch_tables.py <zenodo record id>          the tables (53 files, 40 GB) into tables/redshift/
    python scripts/fetch_tables.py <zenodo record id> year     the results of the year analysis (4 files) into data/year/
"""
import hashlib
import json
import os
import sys
import urllib.request

record = sys.argv[1]
what = sys.argv[2] if len(sys.argv) > 2 else 'tables'
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
target, listing = {'tables': (os.path.join(root, 'tables', 'redshift'), 'tables.sha256'),
                   'year': (os.path.join(root, 'data', 'year'), 'year.sha256')}[what]
os.makedirs(target, exist_ok=True)
manifest = dict(line.split()[::-1] for line in open(os.path.join(root, 'data', 'tables', listing)) if '.npz' in line)
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
print("%s complete: %d files in %s" % (what, len(manifest), target))
