"""OpenAlex field-tagging for the ORCID placement population.

Tags each person (ORCID) with one of OpenAlex's 26 high-level *fields* (Scopus-derived,
under 4 domains), for cutting per-field placement subnetworks and re-validating Wapman
per field. Replaces the noisy free-text `field_text` keyword heuristic.

Two ID-level routes, both keyed on the ORCID (no name matching here):
  Route A  ORCID -> /authors?filter=orcid:o1|o2|...  -> dominant field over author.topics
  Route B  (A-miss) ORCID -> /works?filter=author.orcid:<orcid> -> mode of primary_topic.field
Residual (neither route resolves) is left for a later low-confidence name+affiliation pass.

Auth (2026): API key is a query param `api_key=` (NOT a header); `mailto=` for the polite
pool. Free keyed tier ~ $1/day, 10,000 list+filter calls/day. Key/mailto live in the
gitignored secrets.json (or env OPENALEX_API_KEY / OPENALEX_MAILTO).

All raw author/works lookups are cached to JSONL under data/interim/openalex_cache/ so reruns
and the full-population run resume for free and stay inside the daily budget.
"""
import json, os, time, hashlib
from collections import Counter
from pathlib import Path

import requests

BASE = "https://api.openalex.org"
ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "interim" / "openalex_cache"
OR_BATCH = 50            # OpenAlex OR-filter accepts up to 100; 50 keeps the URL short
PER_PAGE = 200
SLEEP = 0.12             # ~8 req/s, under the documented limit; bumped on 429
TIMEOUT = 30


def _creds():
    f = ROOT / "secrets.json"
    if f.exists():
        d = json.loads(f.read_text())
        key = d.get("openalex_api_key"); mail = d.get("openalex_mailto")
    else:
        key, mail = None, None
    key = os.environ.get("OPENALEX_API_KEY", key)
    mail = os.environ.get("OPENALEX_MAILTO", mail)
    return key, mail


def _session():
    key, mail = _creds()
    s = requests.Session()
    s.params = {}
    if mail:
        s.params["mailto"] = mail
    if key:
        s.params["api_key"] = key
    s.headers["User-Agent"] = f"degrees-of-separation/1.0 (mailto:{mail})"
    return s


def _get(session, path, params, tries=5):
    """GET with polite retry/backoff on 429 and 5xx. Returns parsed JSON or None."""
    for i in range(tries):
        try:
            r = session.get(f"{BASE}{path}", params=params, timeout=TIMEOUT)
        except requests.RequestException:
            time.sleep(1.0 + i); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2.0 * (i + 1)); continue
        # 4xx other than 429 -> unrecoverable for this query
        return None
    return None


# ---------------------------------------------------------------- field logic
def dominant_field(author):
    """Aggregate author.topics (career-cumulative) by field, weighted by topic.count;
    return the modal field display_name. Fall back to level-0 x_concepts if topics empty."""
    c = Counter()
    for t in (author.get("topics") or []):
        f = (t.get("field") or {}).get("display_name")
        if f:
            c[f] += t.get("count", 1) or 1
    if c:
        return c.most_common(1)[0][0]
    zero = [x for x in (author.get("x_concepts") or []) if x.get("level") == 0]
    if zero:
        return max(zero, key=lambda x: x.get("score", 0)).get("display_name")
    return None


def bare(orcid):
    return (orcid or "").rsplit("/", 1)[-1].strip()


# ---------------------------------------------------------------- cache
def _cache_path(kind):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{kind}.jsonl"


def _load_cache(kind):
    p = _cache_path(kind)
    out = {}
    if p.exists():
        with p.open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                out[rec["orcid"]] = rec
    return out


def _append_cache(kind, recs):
    if not recs:
        return
    with _cache_path(kind).open("a") as fh:
        for rec in recs:
            fh.write(json.dumps(rec) + "\n")


# ---------------------------------------------------------------- Route A
def route_a(orcids, session=None, cache=True, log=print):
    """orcids: list of bare ids. Returns {orcid: {'field','works_count'}} for resolved ids.
    Batched OR query, cached per-orcid (resolved AND confirmed-miss are both recorded)."""
    session = session or _session()
    orcids = [bare(o) for o in orcids if o]
    cached = _load_cache("authors") if cache else {}
    todo = [o for o in orcids if o not in cached]
    log(f"  route_a: {len(orcids)} orcids, {len(cached)} cached, {len(todo)} to fetch "
        f"({(len(todo)+OR_BATCH-1)//OR_BATCH} calls)")

    for i in range(0, len(todo), OR_BATCH):
        chunk = todo[i:i + OR_BATCH]
        js = _get(session, "/authors",
                  {"filter": "orcid:" + "|".join(chunk), "per-page": PER_PAGE})
        hit = {}
        if js:
            for a in js.get("results", []):
                oid = bare(a.get("orcid"))
                fld = dominant_field(a); wc = a.get("works_count", 0)
                if fld and (oid not in hit or wc > hit[oid]["works_count"]):
                    hit[oid] = {"orcid": oid, "field": fld, "works_count": wc}
        recs = []
        for o in chunk:
            recs.append(hit.get(o, {"orcid": o, "field": None, "works_count": 0}))
        if cache:
            _append_cache("authors", recs)
        for r in recs:
            cached[r["orcid"]] = r
        time.sleep(SLEEP)
        if (i // OR_BATCH) % 20 == 0 and i:
            log(f"    ... {i}/{len(todo)}")

    return {o: cached[o] for o in orcids
            if o in cached and cached[o].get("field")}


# ---------------------------------------------------------------- Route B
def route_b_one(orcid, session=None, cache_rec=None, max_pages=2):
    """A-miss rescue: mode of primary_topic.field over the author's works (by ORCID).
    Returns {'field','n_works'}. Reads up to max_pages*PER_PAGE works (career is stable;
    a couple hundred works pin the dominant field)."""
    session = session or _session()
    o = bare(orcid)
    fields, n = Counter(), 0
    cursor = "*"
    for _ in range(max_pages):
        js = _get(session, "/works",
                  {"filter": f"author.orcid:{o}", "per-page": PER_PAGE,
                   "cursor": cursor, "select": "primary_topic"})
        if not js:
            break
        for w in js.get("results", []):
            f = ((w.get("primary_topic") or {}).get("field") or {}).get("display_name")
            if f:
                fields[f] += 1; n += 1
        cursor = (js.get("meta") or {}).get("next_cursor")
        if not cursor or len(js.get("results", [])) < PER_PAGE:
            break
        time.sleep(SLEEP)
    if not fields:
        return {"field": None, "n_works": 0}
    return {"field": fields.most_common(1)[0][0], "n_works": n}


def route_b(miss_orcids, session=None, cache=True, log=print):
    """Rescue A-misses one ORCID at a time (cached). Returns {orcid: {'field','n_works'}}."""
    session = session or _session()
    miss = [bare(o) for o in miss_orcids if o]
    cached = _load_cache("works") if cache else {}
    todo = [o for o in miss if o not in cached]
    log(f"  route_b: {len(miss)} A-misses, {len(cached)} cached, {len(todo)} to fetch "
        f"(~{todo and 'up to '+str(len(todo)*2)+' calls' or '0 calls'})")
    for k, o in enumerate(todo):
        rec = route_b_one(o, session=session)
        rec = {"orcid": o, **rec}
        if cache:
            _append_cache("works", [rec])
        cached[o] = rec
        time.sleep(SLEEP)
        if k and k % 100 == 0:
            log(f"    ... {k}/{len(todo)}")
    return {o: cached[o] for o in miss
            if o in cached and cached[o].get("field")}


# ---------------------------------------------------------------- driver
def tag_orcids(orcids, do_route_b=True, log=print):
    """Tag a list of ORCIDs. Returns list of dicts: orcid, field, route, n_works, confidence."""
    orcids = [bare(o) for o in orcids if o]
    a = route_a(orcids, log=log)
    miss = [o for o in orcids if o not in a]
    b = route_b(miss, log=log) if do_route_b else {}
    rows = []
    for o in orcids:
        if o in a:
            rows.append({"orcid": o, "field": a[o]["field"], "route": "A",
                         "n_works": a[o]["works_count"], "confidence": "high"})
        elif o in b:
            rows.append({"orcid": o, "field": b[o]["field"], "route": "B",
                         "n_works": b[o]["n_works"], "confidence": "high"})
        else:
            rows.append({"orcid": o, "field": None, "route": "miss",
                         "n_works": 0, "confidence": "none"})
    return rows
