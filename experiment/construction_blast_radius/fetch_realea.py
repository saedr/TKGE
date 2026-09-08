#!/usr/bin/env python3
import argparse, json, shutil, subprocess, tarfile, zipfile
from pathlib import Path, PurePosixPath

DRIVE_URL = "https://drive.google.com/drive/folders/1x-8OonL8SMDpNyfGyBmwzsgQL_zVMojx?usp=sharing"
ARCHIVE_MAP = {
    "DB-YG-15K": "datasets/main/DBP_en_YG_en_15K_V1/",
    "DB-WD-15K": "datasets/main/DBP_en_WD_en_15K_V1/",
}


def safe_rel(path):
    parts = [x for x in PurePosixPath(path).parts if x not in ("/", ".")]
    if any(x == ".." for x in parts):
        raise RuntimeError("Unsafe path in Drive listing")
    return Path(*parts)


def resolve_listing():
    cp = subprocess.run(
        ["gdown", DRIVE_URL, "--folder", "--json"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=180,
    )
    if cp.returncode != 0:
        raise RuntimeError("gdown folder resolution failed: " + cp.stderr[-2000:])
    return json.loads(cp.stdout)


def download(url, out):
    out.parent.mkdir(parents=True, exist_ok=True)
    cp = subprocess.run(
        ["gdown", url, "-O", str(out)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=900,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"Download failed for {url}: {cp.stdout[-2000:]}")


def has_datasets(root):
    root = Path(root) / "RealEA"
    for name in ARCHIVE_MAP:
        ds = root / name
        required = [
            "rel_triples_1", "rel_triples_2", "attr_triples_1", "attr_triples_2",
            "ent_links", "721_5folds",
        ]
        if not ds.is_dir() or any(not (ds / x).exists() for x in required):
            return False
    return True


def write_archive_listing(dest, names):
    names = [str(n).replace("\\", "/") for n in names]
    (Path(dest) / "archive_listing.txt").write_text("\n".join(names), encoding="utf-8")


def target_for_member(name, dest):
    normalized = name.replace("\\", "/")
    for canonical, prefix in ARCHIVE_MAP.items():
        if normalized.startswith(prefix):
            remainder = normalized[len(prefix):]
            if not remainder or remainder.endswith("/"):
                return None
            return Path(dest) / "RealEA" / canonical / safe_rel(remainder)
    return None


def extract_selected(archive, dest):
    archive = Path(archive)
    extracted = {k: 0 for k in ARCHIVE_MAP}
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            all_names = z.namelist()
            write_archive_listing(dest, all_names)
            for n in all_names:
                target = target_for_member(n, dest)
                if target is None:
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(n) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                for canonical, prefix in ARCHIVE_MAP.items():
                    if n.replace("\\", "/").startswith(prefix):
                        extracted[canonical] += 1
                        break
    else:
        try:
            with tarfile.open(archive) as t:
                members = t.getmembers()
                write_archive_listing(dest, [m.name for m in members])
                for m in members:
                    if not m.isfile():
                        continue
                    target = target_for_member(m.name, dest)
                    if target is None:
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    src = t.extractfile(m)
                    if src:
                        with src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        for canonical, prefix in ARCHIVE_MAP.items():
                            if m.name.replace("\\", "/").startswith(prefix):
                                extracted[canonical] += 1
                                break
        except tarfile.TarError:
            return False
    print("EXTRACTED", json.dumps(extracted, sort_keys=True), flush=True)
    return all(v > 0 for v in extracted.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    dest = Path(args.output)
    dest.mkdir(parents=True, exist_ok=True)

    rows = resolve_listing()
    (dest / "drive_listing.json").write_text(json.dumps(rows, indent=2))
    archives = [
        r for r in rows
        if str(r.get("path", "")).lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
    ]
    if not archives:
        raise RuntimeError("Published RealEA Drive folder contains no downloadable dataset archive")

    archives = sorted(archives, key=lambda r: str(r.get("path", "")))
    success = False
    for i, row in enumerate(archives[:2], 1):
        archive = dest / (f"_archive_{i}_" + Path(str(row["path"])).name)
        print("Downloading dataset archive", row["path"], flush=True)
        download(row["url"], archive)
        if extract_selected(archive, dest):
            success = True
            break

    if not success or not has_datasets(dest):
        raise RuntimeError("Frozen RealEA base 15K datasets were not retrievable from datasets/main in the published archive")
    print("REALEA_DATA_READY")


if __name__ == "__main__":
    main()
