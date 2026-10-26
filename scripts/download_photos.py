"""Download freely licensed photos from Openverse into catdog_photos/.

Fills catdog_photos/{own/neither, holdout/neither, holdout/cat, holdout/dog}
and writes SOURCES.csv so every photo can be credited. The folder sits
outside data/ on purpose: check the photos by eye, then copy the good ones
into data/own/ and data/holdout/. Run:  python scripts/download_photos.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import re
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import requests
from PIL import Image, ImageOps

API_URL = "https://api.openverse.org/v1/images/"
OUT_DIR = Path(__file__).resolve().parent.parent / "catdog_photos"
USER_AGENT = "catdog-capstone-photo-downloader/1.0 (student project; python-requests)"

PAGE_SIZE = 20
MAX_PAGE = 5
API_SLEEP = 1.0       # seconds between Openverse API calls
RETRIES = 3
MIN_SIDE = 300        # px; smaller images are skipped
MAX_SIDE = 1600       # px; the longer side is resized down to this
JPEG_QUALITY = 90

# (folder, file prefix, target count, queries)
PLAN: list[tuple[str, str, int, list[str]]] = [
    ("own/neither", "neither", 60, [
        "Lagos street", "Abuja street", "Nigerian market", "okada motorcycle Nigeria", "jollof rice",
        "Nigerian food plate", "living room interior", "bedroom interior", "plastic chair", "kitchen interior",
        "goat Nigeria", "chicken farm Africa", "people Nigeria street", "classroom Africa", "car traffic Lagos",
        "office desk", "shop front Africa", "sofa living room", "rice dish", "bus station Nigeria",
    ]),
    ("holdout/neither", "neither", 15, [
        "Nigerian street market", "goat", "rooster", "African kitchen", "motorcycle taxi Africa", "fruit stall",
    ]),
    ("holdout/cat", "cat", 12, ["cat indoors", "cat on sofa", "street cat", "kitten home"]),
    ("holdout/dog", "dog", 12, ["dog outdoors", "dog at home", "puppy", "street dog Africa"]),
]

# Extra queries used only by --replace, to find photos the first run did not see.
EXTRA_QUERIES = {
    "cat": ["tabby cat", "cat portrait", "cat sleeping", "cat on windowsill", "domestic cat"],
    "dog": ["dog portrait", "dog walking", "labrador dog", "dog lying down", "domestic dog"],
}

# Whole words only (plus plurals), so "education" or "vacation" do not count as "cat".
PET_WORDS = re.compile(r"\b(cats?|kittens?|dogs?|pupp(y|ies)|pets?|canines?|felines?)\b", re.IGNORECASE)

# For the cat/dog folders: the title or tags must name the animal, and must not name anything that would make the
# photo ambiguous (the other animal, look-alikes) or not a real photo of an animal (artwork).
ART_WORDS = r"drawings?|paintings?|illustrations?|art|artwork|murals?|sketch(es)?|cartoons?|statues?|sculptures?|toys?|graffiti"
ANIMAL_RULES = {
    "cat": (re.compile(r"\b(cats?|kittens?|kitty|felines?|tabby)\b", re.IGNORECASE),
            re.compile(rf"\b(dogs?|pupp(y|ies)|fox(es)?|wolf|wolves|lions?|tigers?|leopards?|{ART_WORDS})\b", re.IGNORECASE)),
    "dog": (re.compile(r"\b(dogs?|pupp(y|ies)|canines?)\b", re.IGNORECASE),
            re.compile(rf"\b(cats?|kittens?|fox(es)?|wolf|wolves|jackals?|{ART_WORDS})\b", re.IGNORECASE)),
}

CSV_COLUMNS = ["file", "folder", "query", "title", "creator", "license", "license_version", "source_url"]

session = requests.Session()
session.headers["User-Agent"] = USER_AGENT
_last_api_call = 0.0


def get_with_retry(url: str, params: dict | None = None, timeout: float = 30) -> requests.Response | None:
    """GET with up to RETRIES retries on HTTP 429/5xx and network errors, with exponential backoff."""
    for attempt in range(RETRIES + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout)
        except requests.RequestException as exc:
            reason = type(exc).__name__
        else:
            if resp.status_code < 400:
                return resp
            if resp.status_code != 429 and resp.status_code < 500:
                return None  # 403/404 etc.: retrying will not help
            reason = f"HTTP {resp.status_code}"
        if attempt < RETRIES:
            wait = 2 ** (attempt + 1)
            print(f"    {reason}, retrying in {wait}s ...")
            time.sleep(wait)
    return None


def search(query: str, photos_only: bool = False) -> Iterator[dict]:
    """Yield Openverse results for a query, fetching pages lazily up to MAX_PAGE."""
    global _last_api_call
    for page in range(1, MAX_PAGE + 1):
        time.sleep(max(0.0, API_SLEEP - (time.monotonic() - _last_api_call)))
        params = {"q": query, "license_type": "all", "mature": "false", "page_size": PAGE_SIZE, "page": page}
        if photos_only:
            params["category"] = "photograph"
        resp = get_with_retry(API_URL, params=params)
        _last_api_call = time.monotonic()
        if resp is None:
            print(f"    [{query}] page {page}: API request failed, moving on")
            return
        data = resp.json()
        results = data.get("results", [])
        yield from results
        if not results or page >= data.get("page_count", 0):
            return


def _words(result: dict) -> list[str]:
    return [result.get("title") or ""] + [t.get("name", "") for t in result.get("tags") or []]


def mentions_pet(result: dict) -> bool:
    return any(PET_WORDS.search(w) for w in _words(result))


def accepts(prefix: str, result: dict) -> bool:
    """Metadata filter applied before downloading."""
    if prefix == "neither":
        return not mentions_pet(result)
    required, rejected = ANIMAL_RULES[prefix]
    words = _words(result)
    return any(required.search(w) for w in words) and not any(rejected.search(w) for w in words)


def fetch_image(result: dict, seen_hashes: set[str]) -> Image.Image | None:
    """Download, dedupe by MD5, open, orient, size-check and resize one result. None = skip it."""
    url = result.get("url")
    if not url:
        return None
    resp = get_with_retry(url)
    if resp is None:
        return None
    digest = hashlib.md5(resp.content).hexdigest()
    if digest in seen_hashes:
        return None
    seen_hashes.add(digest)
    try:
        img = Image.open(io.BytesIO(resp.content))
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:  # anything Pillow cannot open or decode
        return None
    if min(img.size) < MIN_SIDE:
        return None
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
    return img


def fill_folder(folder: str, prefix: str, names: list[str], queries: list[str],
                seen_ids: set[str], seen_hashes: set[str], rows: list[dict], seen_urls: set[str] = frozenset()) -> int:
    """Save one photo per name in `names`, round-robin across queries so downloads spread evenly."""
    dest = OUT_DIR / folder
    dest.mkdir(parents=True, exist_ok=True)
    target = len(names)
    print(f"\n== {folder}: target {target} from {len(queries)} queries")
    streams = {q: search(q, photos_only=prefix != "neither") for q in queries}
    saved = 0
    while saved < target and streams:
        for query in list(streams):
            if saved >= target:
                break
            img = None
            for result in streams[query]:
                if result.get("id") in seen_ids or result.get("foreign_landing_url") in seen_urls:
                    continue
                seen_ids.add(result.get("id"))
                if not accepts(prefix, result):
                    continue
                img = fetch_image(result, seen_hashes)
                if img is not None:
                    break
            if img is None:  # this query has nothing usable left
                del streams[query]
                continue
            name = names[saved]
            saved += 1
            img.save(dest / name, "JPEG", quality=JPEG_QUALITY)
            rows.append({
                "file": name, "folder": folder, "query": query,
                "title": result.get("title") or "", "creator": result.get("creator") or "",
                "license": result.get("license") or "", "license_version": result.get("license_version") or "",
                "source_url": result.get("foreign_landing_url") or "",
            })
            print(f"  [{saved:>2}/{target}] {name}  <- {query}")
    return saved


def replace(folder: str, files: list[str]) -> int:
    """Re-download the given files in one folder with new photos, and update their SOURCES.csv rows.

    Every photo already listed in SOURCES.csv, or rejected before (REJECTED.txt), is skipped.
    """
    plan = {f: (prefix, queries) for f, prefix, _, queries in PLAN}
    if folder not in plan:
        print(f"Unknown folder {folder!r}. Choose from {sorted(plan)}")
        return 1
    prefix, queries = plan[folder]
    names = [f if f.endswith(".jpg") else f"{f}.jpg" for f in files]
    csv_path = OUT_DIR / "SOURCES.csv"
    with open(csv_path, newline="", encoding="utf-8") as f:
        old_rows = list(csv.DictReader(f))
    rejected_path = OUT_DIR / "REJECTED.txt"
    rejected = rejected_path.read_text(encoding="utf-8").split() if rejected_path.exists() else []
    seen_urls = {r["source_url"] for r in old_rows} | set(rejected)
    # Remember the photos being replaced, so a later --replace never brings them back.
    with open(rejected_path, "a", encoding="utf-8") as f:
        f.writelines(r["source_url"] + "\n" for r in old_rows if r["folder"] == folder and r["file"] in names)

    new_rows: list[dict] = []
    saved = fill_folder(folder, prefix, names, queries + EXTRA_QUERIES.get(prefix, []),
                        set(), set(), new_rows, seen_urls)
    replaced = {r["file"]: r for r in new_rows}
    rows = [replaced.get(r["file"], r) if r["folder"] == folder else r for r in old_rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nReplaced {saved} of {len(names)} in {folder}.")
    missing = [n for n in names if n not in replaced]
    if missing:
        print(f"{len(missing)} missing (old photo left in place): {', '.join(missing)}")
    return 0


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # Windows consoles choke on some photo titles
    if len(sys.argv) > 1:
        if sys.argv[1] != "--replace" or len(sys.argv) < 4:
            print("Usage: python scripts/download_photos.py [--replace FOLDER FILE ...]\n"
                  "   e.g. python scripts/download_photos.py --replace holdout/cat cat_001 cat_004")
            return 1
        return replace(sys.argv[2], sys.argv[3:])

    existing = [p for p in OUT_DIR.rglob("*.jpg")] if OUT_DIR.exists() else []
    if existing:
        print(f"{OUT_DIR} already has {len(existing)} photos. Move or delete it first so nothing is overwritten.")
        return 1

    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    rows: list[dict] = []
    counts = {}
    for folder, prefix, target, queries in PLAN:
        names = [f"{prefix}_{i:03d}.jpg" for i in range(1, target + 1)]
        counts[folder] = (fill_folder(folder, prefix, names, queries, seen_ids, seen_hashes, rows), target)

    with open(OUT_DIR / "SOURCES.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{'folder':<18}{'saved':>6}{'target':>8}  note")
    for folder, (saved, target) in counts.items():
        note = f"{target - saved} missing" if saved < target else "ok"
        print(f"{folder:<18}{saved:>6}{target:>8}  {note}")
    print(f"\nCredits for all {len(rows)} photos: {OUT_DIR / 'SOURCES.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
