#!/usr/bin/env python3
"""Unified issue cache for the Triton upgrade report.

Maintains a single local snapshot of ALL intel/torch-xpu-ops issues at
    ~/torch-xpu-ops.issues        (JSON)

Behaviour (see sync()):
  * first run (no cache)  -> fetch every issue (state=all) and save.
  * later runs (cache)    -> incremental: fetch only issues updated since the
                            last sync (new issues + any that changed state /
                            body, e.g. an opened issue that got closed), merge,
                            and re-save.

The report generator (gen_report.py) imports this module, calls sync() once, and
then matches each section against the cached issues (UT by case keys; accuracy /
performance / PT2E by error-message similarity).

Standard library only. Set GITHUB_TOKEN to raise API rate limits (optional).
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

REPO = "intel/torch-xpu-ops"
CACHE_PATH = os.path.expanduser("~/torch-xpu-ops.issues")
DIR_PATH = os.path.expanduser("~/torch-xpu-ops.issues.dir")   # one JSON per issue

# ---- error-signature extraction (shared with the report) ------------------- #
ERR_RE = re.compile(
    r"^(.*?(?:[A-Za-z_][\w.]*(?:Error|Exception|Warning)|"
    r"out of memory|OutOfMemoryError|assert\w*|RuntimeError|NotImplementedError|"
    r"UR_RESULT_ERROR\w*)\b.*)$",
    re.IGNORECASE,
)
NOISE_RE = re.compile(r"traceback|most recent call last|file \"|warnings?\.warn", re.IGNORECASE)
CLASS_RE = re.compile(r"^[A-Za-z0-9_./-]+$")


def _clean(line):
    s = line.strip().strip("`").strip()
    s = re.sub(r"^[WEI]\d{4}\s+[\d:.]+\s+\d+\s+", "", s)  # log timestamps
    return s.strip()


def extract_signatures(title, body):
    """Return de-duplicated error-message signatures for an issue."""
    sigs, seen = [], set()

    def add(sig):
        sig = _clean(sig)
        if len(sig) < 12 or NOISE_RE.search(sig):
            return
        key = re.sub(r"\d+", "", sig.lower())
        if key in seen:
            return
        seen.add(key)
        sigs.append(sig)

    m = re.search(r"((?:[A-Za-z_][\w.]*(?:Error|Exception))\b.*)$", title or "")
    if m:
        add(m.group(1))
    for raw in (body or "").replace("\r", "").split("\n"):
        s = _clean(raw)
        if not s or s.startswith("#") or s.startswith("|"):
            continue
        mm = ERR_RE.match(s)
        if mm:
            add(mm.group(1))
    return sigs[:6]


def parse_cases(body):
    """Return [(cls, name, fixed)] from a 'Cases:' block in an issue body.

    Accepts both `class,name` and `category,class,name` line forms, mixed within
    one block. The class may be bare or a full module path; an optional
    `~~...~~` strike-through marks a fixed case.
    """
    if not body:
        return []
    cases, started = [], False
    for raw in body.replace("\r", "").split("\n"):
        s = raw.strip()
        if not started:
            if s.lower().startswith("cases:"):
                started = True
                rest = s[len("cases:"):].strip()
                if not rest:
                    continue
                s = rest
            else:
                continue
        if s == "" or s.startswith("#") or s.startswith("```") or s.lower().startswith("last good"):
            break
        fixed = s.startswith("~~") and s.endswith("~~")
        core = s[2:-2].strip() if fixed else s
        parts = core.split(",")
        if len(parts) == 2:                      # class,name
            cls, name = parts[0].strip(), parts[1].strip()
        elif len(parts) >= 3:                    # category,class,name (class empty
            cls, name = parts[1].strip(), ",".join(parts[2:]).strip()  # for a
        else:                                    # collection-failure file entry)
            break
        if name and " " not in name and (cls == "" or CLASS_RE.match(cls)):
            cases.append((cls, name, fixed))
        else:
            break
    return cases


UT_NODEID_RE = re.compile(
    r"[A-Za-z0-9_./\\-]*?[A-Za-z0-9_-]+\.py::"   # a *.py test-file path
    r"([A-Za-z_]\w*)::"                          # test class
    r"(test\w*(?:\[[^\]\r\n]*\])?)"             # test name (+ optional [params])
)
UT_BULLET_CLASS_RE = re.compile(
    r"^\*\*(?:`)?([A-Za-z0-9_./\\-]+\.py)(?:`)?\s*[-—]\s*([A-Za-z_]\w*)\*\*$"
)
UT_BULLET_TEST_RE = re.compile(
    r"^\s*[-*]\s*(?:`)?(test\w*(?:\[[^\]\r\n]*\])?)(?:`)?\s*$"
)


def extract_ut_nodeids(text):
    """Return [(cls, name)] for pytest node ids and markdown test lists found
    free-form in an issue title/body. Used to match UT failures against issues
    that describe cases in prose (title / 'Affected Test Cases') with no
    structured `Cases:` block, e.g. #4253 and #4947.
    """
    out, seen = [], set()

    # Normal pytest node ids: file.py::Class::test_name
    for m in UT_NODEID_RE.finditer(text or ""):
        key = (m.group(1), m.group(2))
        if key not in seen:
            seen.add(key)
            out.append(key)

    # Markdown sections like:
    #   **`test_indexing_xpu.py` — TestIndexingXPU**
    #   - `test_index_add_fast_path_xpu_float64`
    current_cls = None
    for raw in (text or "").replace("\r", "").split("\n"):
        s = raw.strip()
        if not s:
            current_cls = None
            continue
        m = UT_BULLET_CLASS_RE.match(s)
        if m:
            current_cls = m.group(2)
            continue
        if current_cls is None:
            continue
        mm = UT_BULLET_TEST_RE.match(s)
        if not mm:
            continue
        key = (current_cls, mm.group(1))
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


# ---- GitHub API ------------------------------------------------------------ #
def _api_get(url):
    headers = {"User-Agent": "xpu-issue-cache", "Accept": "application/vnd.github+json"}
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=30) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code in (403, 429) and attempt < 3:
                time.sleep(20)
                continue
            raise
    raise RuntimeError("unreachable")


def _normalize(it):
    return {
        "number": it["number"],
        "title": it.get("title", "") or "",
        "body": it.get("body", "") or "",
        "state": it.get("state", "open"),
        "labels": [l["name"] for l in it.get("labels", [])],
        "url": it.get("html_url", ""),
        "created_at": it.get("created_at", ""),
        "user": (it.get("user") or {}).get("login", ""),
        "updated_at": it.get("updated_at", ""),
        "comments_count": it.get("comments", 0) or 0,
        "comments": "",   # concatenated comment bodies (filled by _enrich_comments)
    }


def _fetch_comments_text(number):
    """Return all comment bodies for an issue joined by newlines ('' on failure).

    Fetches up to the first few pages of comments (per_page=100). Used so that a
    case can be considered 'already tracked' when its key appears in a comment,
    not just the issue title/body.
    """
    parts = []
    page = 1
    try:
        while page <= 5:  # cap at 500 comments; enough for tracking issues
            url = (f"https://api.github.com/repos/{REPO}/issues/{number}/comments"
                   f"?per_page=100&page={page}")
            data = _api_get(url)
            if not data:
                break
            for c in data:
                b = c.get("body") or ""
                if b:
                    parts.append(b)
            if len(data) < 100:
                break
            page += 1
    except (urllib.error.URLError, RuntimeError):
        return "\n".join(parts)
    return "\n".join(parts)


def _enrich_comments(iss, prev=None, verbose=False):
    """Populate iss['comments'] with comment bodies when the issue has any.

    Reuses previously cached comment text when the issue hasn't gained comments
    since the last sync, to avoid re-fetching unchanged discussions.
    """
    count = iss.get("comments_count", 0) or 0
    if count <= 0:
        iss["comments"] = ""
        return iss
    if (prev is not None and prev.get("comments")
            and prev.get("comments_count", -1) == count):
        iss["comments"] = prev.get("comments", "")
        return iss
    if verbose:
        print(f"    fetching {count} comment(s) for #{iss['number']}")
    iss["comments"] = _fetch_comments_text(iss["number"])
    return iss


def _iter_updated(since_start=None, verbose=True):
    """Yield issues (excluding PRs) via a `since` cursor to dodge the offset cap.

    Walks updated_at ascending, advancing `since` to the last page's newest
    timestamp. Callers de-duplicate by issue number. since_start=None fetches all.
    """
    since = since_start
    seen = set()
    while True:
        since_q = f"&since={since}" if since else ""
        url = (f"https://api.github.com/repos/{REPO}/issues"
               f"?state=all&per_page=100&sort=updated&direction=asc{since_q}&page=1")
        batch = _api_get(url)
        if not batch:
            break
        page_new, last = 0, since
        for it in batch:
            last = it.get("updated_at") or last
            if "pull_request" in it:
                continue
            if it["number"] not in seen:
                seen.add(it["number"])
                page_new += 1
            yield _normalize(it)
        if verbose:
            print(f"    +{page_new} new (total seen {len(seen)}) up to {last}")
        if len(batch) < 100 or page_new == 0 or last == since:
            break
        since = last
        time.sleep(1)



def load():
    """Return the cache dict {'repo','synced_at','issues':{num:issue}} or None."""
    if not os.path.isfile(CACHE_PATH):
        return None
    try:
        with open(CACHE_PATH) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _save(cache):
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    os.replace(tmp, CACHE_PATH)


def _search_seed(verbose=True, pages=6):
    """Fallback seed via the Search API (separate quota) when the core issues
    endpoint is rate-limited. Grabs the most-recently-updated issues (partial)."""
    issues = {}
    for p in range(1, pages + 1):
        url = (f"https://api.github.com/search/issues?q=repo:{REPO}+is:issue"
               f"&sort=updated&order=desc&per_page=100&page={p}")
        data = _api_get(url)
        batch = data.get("items", [])
        for it in batch:
            if "pull_request" in it:
                continue
            issues[str(it["number"])] = _normalize(it)
        if verbose:
            print(f"    search page {p}: {len(issues)} issues so far")
        if len(batch) < 100:
            break
        time.sleep(6)  # unauthenticated search limit ~10/min
    return issues



def sync(full=False, verbose=True, offline_ok=True):
    """Ensure the local cache is present and current; return the cache dict.

    * no cache            -> full fetch (issues endpoint, `since` walk).
    * cache marked partial-> retry a full fetch to complete it.
    * cache complete      -> incremental fetch of issues updated since last sync.

    If the core issues endpoint is rate-limited / offline, falls back to an
    existing cache, or seeds a partial cache from the Search API, or (last
    resort, offline_ok) returns an empty cache so the report can still render.
    """
    cache = None if full else load()
    started = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    incremental = bool(cache and cache.get("issues") and not cache.get("partial"))
    try:
        if incremental:
            since = cache.get("synced_at")
            if verbose:
                print(f"  Updating issue cache since {since} ...")
            issues = cache["issues"]
            n_new = 0
            for iss in _iter_updated(since_start=since, verbose=verbose):
                key = str(iss["number"])
                if key not in issues:
                    n_new += 1
                _enrich_comments(iss, prev=issues.get(key), verbose=verbose)
                issues[key] = iss
            cache["synced_at"] = started
            cache.pop("partial", None)
            _save(cache)
            if verbose:
                print(f"  Cache updated: +{n_new} new, {len(issues)} total -> {CACHE_PATH}")
        else:
            if verbose:
                print(f"  Fetching ALL {REPO} issues ...")
            issues = dict(cache["issues"]) if cache and cache.get("issues") else {}
            for iss in _iter_updated(since_start=None, verbose=verbose):
                _enrich_comments(iss, prev=issues.get(str(iss["number"])), verbose=verbose)
                issues[str(iss["number"])] = iss
            cache = {"repo": REPO, "synced_at": started, "issues": issues}
            _save(cache)
            if verbose:
                print(f"  Cache complete: {len(issues)} issues -> {CACHE_PATH}")
        return cache
    except (urllib.error.URLError, RuntimeError) as e:
        existing = load()
        if existing and existing.get("issues"):
            if verbose:
                print(f"  WARNING: issue sync failed ({e}); using existing cache "
                      f"({len(existing['issues'])} issues).")
            return existing
        # No usable cache — try the Search API as a partial seed.
        try:
            if verbose:
                print(f"  Core issues endpoint unavailable ({e}); seeding via Search API ...")
            seed = _search_seed(verbose=verbose)
            if seed:
                cache = {"repo": REPO, "synced_at": "1970-01-01T00:00:00Z",
                         "issues": seed, "partial": True}
                _save(cache)
                if verbose:
                    print(f"  Seeded {len(seed)} issues (partial) -> {CACHE_PATH}")
                return cache
        except (urllib.error.URLError, RuntimeError) as e2:
            if verbose:
                print(f"  Search seed also failed ({e2}).")
        if offline_ok:
            if verbose:
                print("  Proceeding without issue matches.")
            return {"repo": REPO, "synced_at": started, "issues": {}}
        raise


def _iter_all_pages(verbose=True):
    """Yield every issue (excluding PRs) via plain page pagination (state=all).

    Unlike the `since`-cursor walk used by sync(), this visits every page in
    order so no issue is skipped. Safe because the repo has well under the
    ~10000-item offset cap.
    """
    page = 1
    while True:
        url = (f"https://api.github.com/repos/{REPO}/issues"
               f"?state=all&per_page=100&sort=created&direction=asc&page={page}")
        batch = _api_get(url)
        if not batch:
            break
        kept = 0
        for it in batch:
            if "pull_request" in it:
                continue
            kept += 1
            yield _normalize(it)
        if verbose:
            print(f"    page {page}: {len(batch)} items ({kept} issues)")
        if len(batch) < 100:
            break
        page += 1
        time.sleep(0.3)


def _dir_latest_updated(dir_path):
    """Newest issue updated_at across the per-issue JSONs ('' when empty)."""
    latest = ""
    if os.path.isdir(dir_path):
        for fn in os.listdir(dir_path):
            if not fn.endswith(".json"):
                continue
            try:
                with open(os.path.join(dir_path, fn)) as fh:
                    u = json.load(fh).get("updated_at", "")
            except (OSError, ValueError):
                continue
            if u > latest:
                latest = u
    return latest


def _write_issue_json(dir_path, iss, prev=None):
    """Write one issue JSON, preserving previously-fetched comments (from the dir
    file, else the monolithic cache) when the fresh copy has none."""
    key = str(iss["number"])
    path = os.path.join(dir_path, key + ".json")
    if not iss.get("comments"):
        cmt = ""
        if os.path.isfile(path):
            try:
                with open(path) as fh:
                    cmt = json.load(fh).get("comments", "")
            except (OSError, ValueError):
                cmt = ""
        if not cmt and prev:
            cmt = (prev.get(key) or {}).get("comments", "")
        if cmt:
            iss["comments"] = cmt
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(iss, fh)
    os.replace(tmp, path)


def sync_dir(dir_path=DIR_PATH, full=False, enrich_from=CACHE_PATH, verbose=True):
    """Write one JSON per issue into `dir_path` (gap-free per-issue cache).

    Incremental by default: fetches only issues updated since the newest cached
    `updated_at`, so newly filed issues (e.g. a just-created bug) are picked up
    cheaply. `full=True`, or an empty dir, does a complete page walk. Returns the
    number of issue files written/updated.
    """
    os.makedirs(dir_path, exist_ok=True)
    prev = {}
    if enrich_from and os.path.isfile(enrich_from):
        try:
            with open(enrich_from) as fh:
                prev = json.load(fh).get("issues", {})
        except (OSError, ValueError):
            prev = {}
    since = "" if full else _dir_latest_updated(dir_path)
    n = 0
    if since:
        if verbose:
            print(f"  Updating issue dir since {since} -> {dir_path}")
        for iss in _iter_updated(since_start=since, verbose=verbose):
            _write_issue_json(dir_path, iss, prev)
            n += 1
    else:
        if verbose:
            print(f"  Fetching ALL {REPO} issues (page walk) -> {dir_path}")
        for iss in _iter_all_pages(verbose=verbose):
            _write_issue_json(dir_path, iss, prev)
            n += 1
    if verbose:
        print(f"  Wrote/updated {n} issue JSONs -> {dir_path}")
    return n


def load_dir(dir_path=DIR_PATH):
    """Return the issue list from a per-issue JSON directory ([] if absent)."""
    if not os.path.isdir(dir_path):
        return []
    out = []
    for fn in os.listdir(dir_path):
        if not fn.endswith(".json"):
            continue
        try:
            with open(os.path.join(dir_path, fn)) as fh:
                out.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Sync intel/torch-xpu-ops issue cache.")
    ap.add_argument("--full", action="store_true", help="force full re-fetch")
    ap.add_argument("--dir", action="store_true",
                    help="write one JSON per issue into "
                         f"{DIR_PATH} (incremental; add --full for a page walk)")
    args = ap.parse_args()
    if args.dir:
        sync_dir(full=args.full, verbose=True)
    else:
        sync(full=args.full, verbose=True)


if __name__ == "__main__":
    main()
