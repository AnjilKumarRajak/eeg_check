
import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.error
import urllib.request

OSF_NODES = {"v1": "q3zws", "v2": "2urht"}
WANTED = [
    ("v1", "task1", "SR",  "task1-SR"),
    ("v1", "task2", "NR",  "task2-NR"),
    ("v2", "task1", "NR",  "task2-NR-2.0"),
    ("v1", "task3", "TSR", "task3-TSR"),
]


def api_get(url, retries=5, page_size=None):
    if page_size and "page%5Bsize%5D" not in url and "page[size]" not in url:
        url += ("&" if "?" in url else "?") + f"page[size]={page_size}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except Exception as e:
            if attempt == retries - 1:
                raise
            print(f"  API retry {attempt+1}/{retries} after {e}", flush=True)
            time.sleep(2 * (attempt + 1))


def walk_files(url, depth=0, maxdepth=3):
    while url:
        d = api_get(url, page_size=100)
        for it in d.get("data", []):
            a = it["attributes"]
            if a["kind"] == "file":
                if a["name"].endswith(".mat") and "atlab" in a["materialized_path"]:
                    yield a["materialized_path"], (a.get("size") or 0), it["links"]["download"]
            elif depth < maxdepth:
                norm = a["name"].lower()
                inside_matlab = "atlab" in a["materialized_path"]
                if not (inside_matlab or "matlab" in norm.replace(" ", "")):
                    continue  # prune: cannot contain Matlab files
                sub = it["relationships"]["files"]["links"]["related"]["href"]
                yield from walk_files(sub, depth + 1, maxdepth)
        url = d.get("links", {}).get("next")


def find_task_folder(node, task_tok, kind_tok):
    """Locate the top-level OSF folder for a task (names have inconsistent spacing)."""
    root = f"https://api.osf.io/v2/nodes/{node}/files/osfstorage/"
    for it in api_get(root)["data"]:
        a = it["attributes"]
        if a["kind"] != "folder":
            continue
        norm = a["name"].lower().replace(" ", "").replace("-", "")
        if task_tok in norm and kind_tok.lower() in norm:
            return a["name"], it["relationships"]["files"]["links"]["related"]["href"]
    return None, None


def download(url, dest, expected_size):
    part = dest + ".part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    if done and expected_size and done >= expected_size:
        os.rename(part, dest)
        return "resumed-complete"

    for attempt in range(6):
        try:
            req = urllib.request.Request(url)
            if done:
                req.add_header("Range", f"bytes={done}-")
            with urllib.request.urlopen(req, timeout=120) as r, open(part, "ab" if done else "wb") as f:
                t0, last = time.time(), done
                while True:
                    chunk = r.read(4 * 1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    done += len(chunk)
                    if time.time() - t0 > 30:
                        rate = (done - last) / (time.time() - t0) / 1e6
                        pct = f"{100*done/expected_size:5.1f}%" if expected_size else "  ?  "
                        print(f"    {os.path.basename(dest)[:24]:24} {pct} {done/1e9:6.2f} GB {rate:5.1f} MB/s", flush=True)
                        t0, last = time.time(), done
            os.rename(part, dest)
            return "ok"
        except Exception as e:
            done = os.path.getsize(part) if os.path.exists(part) else 0
            print(f"    retry {attempt+1}/6 at {done/1e9:.2f} GB after {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    return "failed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True, help="root dir; files land under <dest>/ZuCo/<task>/Matlab_files")
    ap.add_argument("--tasks", default="task1-SR,task2-NR,task2-NR-2.0",
                    help="comma-separated local task names to fetch")
    ap.add_argument("--workers", type=int, default=4, help="concurrent file downloads")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    want = {t.strip() for t in args.tasks.split(",") if t.strip()}
    ok = skip = fail = 0
    grand_bytes = 0

    # Enumerate and download one task at a time so transfers start immediately.
    for node_key, task_tok, kind_tok, local in WANTED:
        if local not in want:
            continue
        node = OSF_NODES[node_key]
        name, href = find_task_folder(node, task_tok, kind_tok)
        if not href:
            print(f"!! could not locate {task_tok}/{kind_tok} on osf.io/{node}", flush=True)
            continue
        outdir = os.path.join(args.dest, "ZuCo", local, "Matlab_files")
        print(f"\n=== {local}   <- osf.io/{node} '{name}'", flush=True)

        jobs = [(os.path.join(outdir, os.path.basename(mp)), url, size)
                for mp, size, url in walk_files(href)]
        task_bytes = sum(j[2] for j in jobs)
        grand_bytes += task_bytes
        print(f"    {len(jobs)} .mat files, {task_bytes/1e9:.2f} GB", flush=True)
        if args.dry_run:
            continue

        os.makedirs(outdir, exist_ok=True)
        todo = []
        for dest, url, size in sorted(jobs):
            if os.path.exists(dest) and (not size or os.path.getsize(dest) == size):
                skip += 1
            else:
                todo.append((dest, url, size))

        # OSF throttles per-connection, so fetch several files concurrently.
        def fetch(job):
            dest, url, size = job
            print(f"  [{local}] start {os.path.basename(dest)} ({size/1e9:.2f} GB)", flush=True)
            return dest, download(url, dest, size)

        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            for dest, res in pool.map(fetch, todo):
                if res == "failed":
                    fail += 1
                    print(f"    FAILED: {dest}", flush=True)
                else:
                    ok += 1
                    print(f"  [{local}] done  {os.path.basename(dest)}", flush=True)
        print(f"  -- {local} complete", flush=True)

    print(f"\nTOTAL enumerated: {grand_bytes/1e9:.2f} GB", flush=True)
    if args.dry_run:
        return 0
    print(f"DONE  downloaded={ok} skipped={skip} failed={fail}", flush=True)
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
