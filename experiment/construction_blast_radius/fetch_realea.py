#!/usr/bin/env python3
import argparse, json, shutil, subprocess, tarfile, zipfile
from pathlib import Path, PurePosixPath

DRIVE_URL = "https://drive.google.com/drive/folders/1x-8OonL8SMDpNyfGyBmwzsgQL_zVMojx?usp=sharing"
WANTED = ("RealEA/DB-YG-15K/", "RealEA/DB-WD-15K/")

def safe_rel(path):
    parts = [x for x in PurePosixPath(path).parts if x not in ("/", ".")]
    if any(x == ".." for x in parts): raise RuntimeError("Unsafe path in Drive listing")
    return Path(*parts)

def resolve_listing():
    cp = subprocess.run(["gdown", DRIVE_URL, "--folder", "--json"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
    if cp.returncode != 0: raise RuntimeError("gdown folder resolution failed: " + cp.stderr[-2000:])
    return json.loads(cp.stdout)

def download(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run(["gdown", url, "-O", str(out)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=900)
    if cp.returncode != 0: raise RuntimeError(f"Download failed for {url}: {cp.stdout[-2000:]}")

def has_datasets(root):
    roots = [p for p in Path(root).rglob("rel_triples_1") if p.parent.name in {"DB-YG-15K", "DB-WD-15K"} and p.parent.parent.name == "RealEA"]
    return {p.parent.name for p in roots} == {"DB-YG-15K", "DB-WD-15K"}

def extract_selected(archive, dest):
    def wanted(name):
        name = name.replace("\\", "/")
        return any(w in name for w in WANTED)
    archive = Path(archive)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            names = [n for n in z.namelist() if wanted(n)]
            if not names: return False
            for n in names:
                if n.endswith("/"): continue
                target = dest / safe_rel(n); target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(target, "wb") as dst: shutil.copyfileobj(src, dst)
        return True
    try:
        with tarfile.open(archive) as t:
            members = [m for m in t.getmembers() if m.isfile() and wanted(m.name)]
            if not members: return False
            for m in members:
                target = dest / safe_rel(m.name); target.parent.mkdir(parents=True, exist_ok=True)
                src = t.extractfile(m)
                if src:
                    with src, open(target, "wb") as dst: shutil.copyfileobj(src, dst)
        return True
    except tarfile.TarError:
        return False

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output", required=True); args = ap.parse_args()
    dest = Path(args.output); dest.mkdir(parents=True, exist_ok=True)
    rows = resolve_listing(); (dest / "drive_listing.json").write_text(json.dumps(rows, indent=2))
    selected = [r for r in rows if any(w in str(r.get("path", "")).replace("\\", "/") for w in WANTED)]
    if selected:
        for i, row in enumerate(selected, 1):
            rel = safe_rel(row["path"]); print(f"Download {i}/{len(selected)} {rel}", flush=True); download(row["url"], dest / rel)
    else:
        archives = [r for r in rows if str(r.get("path", "")).lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))]
        archives = sorted(archives, key=lambda r: (0 if "entity" in str(r.get("path", "")).lower() else 1, str(r.get("path", ""))))
        for i, row in enumerate(archives[:2], 1):
            archive = dest / (f"_archive_{i}_" + Path(str(row["path"])).name)
            print("Downloading dataset archive", row["path"], flush=True); download(row["url"], archive)
            if extract_selected(archive, dest): break
    if not has_datasets(dest):
        raise RuntimeError("Required public RealEA 15K files were not retrievable from the current published Drive folder.")
    print("REALEA_DATA_READY")

if __name__ == "__main__": main()
