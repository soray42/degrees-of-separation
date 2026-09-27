"""
scripts/77_italy_feasibility_counts.py
=============================================================================
Counts of what exists, for the local write-up ITALY_FEASIBILITY.md. No analysis: no ranks, no
correlations, no earnings figures. Every number the write-up cites is printed here and written to
data/interim/italy_feasibility_counts.csv.

Inputs (public; provenance in data/raw/SOURCES.md, section 10j):
  - ORCID affiliation-transition edges, data/orcid/all/edge_aff/*.parquet (already extracted from
    data/orcid/20260419.7z; nothing is extracted here). The PhD -> faculty filter is scripts/40's,
    imported unchanged (DOC / FAC regexes, one edge per person, earliest faculty-start year).
  - ROR dump data/raw/ror/ror-data.zip (organisation types; denominator of Italian universities).
  - data/raw/italy/almalaurea/: query-tool dropdowns (6 survey years), one result page
    (Bologna, LM biennale, 5 years, by degree class), the 2025 methodology notes, the member list.
  - data/raw/italy/catalog/dati_gov_it_mur_packages.json: the national open-data catalogue record
    of the MUR / USTAT datasets (the USTAT download host refused connections from this machine).
  - data/raw/italy/medicina_anon/: two index pages and one PDF of the anonymous national medicine
    admission-test results.

Deterministic (no randomness). Run: `python scripts/77_italy_feasibility_counts.py`.
"""
from __future__ import annotations

import glob
import html
import importlib.util
import json
import re
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw" / "italy"
OUT = ROOT / "data" / "interim" / "italy_feasibility_counts.csv"

# scripts/40 filters, imported unchanged
_spec = importlib.util.spec_from_file_location("s40", ROOT / "scripts" / "40_crossnational_uk.py")
s40 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s40)
DOC, FAC, SHARDS = s40.DOC, s40.FAC, s40.SHARDS

# Italian-language additions (this script only). "professore" is already matched by FAC ("professor").
DOC_IT = re.compile(r"dottorat|dottore di ricerca|dottoressa di ricerca|dott\.? di ricerca", re.I)
FAC_IT = re.compile(r"\bdocente|ricercat(?:ore|rice|ori)\s+(?:a\s+tempo\s+determinato|universitari|di\s+tipo)"
                    r"|\brtd\s*-?\s*[ab]?\b|\brtt\b", re.I)
RES = re.compile(r"researcher|ricercat(?:ore|rice|ori)", re.I)
DEGMIN = 5  # as scripts/62

ROWS: list[dict] = []


def put(section: str, quantity: str, value, note: str = ""):
    ROWS.append(dict(section=section, quantity=quantity, value=value, note=note))
    print(f"[{section}] {quantity}: {value}" + (f"   ({note})" if note else ""))


# ----------------------------------------------------------------------------------------------
# (b) ORCID: Italian vs British PhD -> faculty edges
# ----------------------------------------------------------------------------------------------
def ror_types():
    z = zipfile.ZipFile(ROOT / "data" / "raw" / "ror" / "ror-data.zip")
    csv = [n for n in z.namelist() if n.endswith(".csv")][0]
    r = pd.read_csv(z.open(csv), usecols=["id", "types", "locations.geonames_details.country_code", "status"],
                    dtype=str)
    return r


def orcid_edges():
    cols = ["role_type_from", "role_type_to", "role_from", "role_to", "org_country_from", "org_country_to",
            "org_from_ror_id", "org_to_ror_id", "org_dept_from", "org_dept_to", "epi_start_year_to",
            "person_orcid"]
    parts = []
    for f in SHARDS:
        d = pd.read_parquet(f, columns=cols)
        d = d[(d.role_type_from == "education") & (d.role_type_to == "employment") &
              d.org_country_from.isin(["it", "gb"]) & (d.org_country_from == d.org_country_to) &
              d.org_from_ror_id.notna() & d.org_to_ror_id.notna()]
        if len(d):
            parts.append(d)
    e = pd.concat(parts, ignore_index=True)
    e["year"] = pd.to_numeric(e.epi_start_year_to, errors="coerce")
    rf, rt = e.role_from.fillna(""), e.role_to.fillna("")
    e["doc"] = rf.str.contains(DOC)
    e["doc_it"] = e.doc | rf.str.contains(DOC_IT)
    e["fac"] = rt.str.contains(FAC)
    e["fac_it"] = e.fac | rt.str.contains(FAC_IT)
    e["res"] = rt.str.contains(RES)
    # rows whose DOC match rests only on the word "doctor" (e.g. "medical doctor"), not a PhD title
    e["doc_only_doctor"] = e.doc & ~rf.str.contains(re.compile(
        r"ph\.?\s?d|d\.phil|dphil|sc\.?d|doctoral|doctorate|doctor of philosophy|philosophiae|dottorat|dottore di ricerca", re.I))
    return e


def dedup(d):
    # scripts/40: one edge per person, earliest faculty-start year
    return d.sort_values("year", kind="mergesort").drop_duplicates("person_orcid", keep="first")


def describe(sec, label, d, univ_ids):
    d = dedup(d)
    deg = pd.concat([d.org_from_ror_id, d.org_to_ror_id]).value_counts()
    nodes = deg.index
    put(sec, f"{label}: edges (one per person)", int(len(d)))
    put(sec, f"{label}: distinct institutions (ROR ids, either end)", int(len(nodes)))
    put(sec, f"{label}: institutions with degree >= {DEGMIN}", int((deg >= DEGMIN).sum()), "degree = in+out, as scripts/62")
    put(sec, f"{label}: of which ROR type 'education'", int(sum(1 for n in nodes[deg >= DEGMIN] if n in univ_ids)))
    put(sec, f"{label}: self-loops (PhD and faculty job at the same institution)", int((d.org_from_ror_id == d.org_to_ror_id).sum()))
    put(sec, f"{label}: between-institution edges", int((d.org_from_ror_id != d.org_to_ror_id).sum()))
    put(sec, f"{label}: edges with a non-empty destination department string",
        int(d.org_dept_to.fillna("").str.strip().ne("").sum()))
    put(sec, f"{label}: edges with faculty-start year >= 2010", int((d.year >= 2010).sum()))
    return d


def section_b():
    sec = "b"
    r = ror_types()
    edu = r[r.types.fillna("").str.contains("education")]
    univ_it = set(edu.loc[edu["locations.geonames_details.country_code"] == "IT", "id"])
    univ_gb = set(edu.loc[edu["locations.geonames_details.country_code"] == "GB", "id"])
    put(sec, "ROR records, country IT, type education (any status)", len(univ_it))
    put(sec, "ROR records, country GB, type education (any status)", len(univ_gb))

    put(sec, "ORCID edge shards read", len(SHARDS), "data/orcid/all/edge_aff; the 7z archive is not opened")
    e = orcid_edges()
    for c, uni in [("gb", univ_gb), ("it", univ_it)]:
        x = e[e.org_country_from == c]
        put(sec, f"{c.upper()}->{c.upper()} education->employment transitions (raw rows)", int(len(x)))
        house = describe(sec, f"{c.upper()} house filter (scripts/40 DOC & FAC)", x[x.doc & x.fac], uni)
        put(sec, f"{c.upper()} house filter: edges whose PhD match rests only on 'doctor'",
            int(house.doc_only_doctor.sum()), "e.g. 'medical doctor'")
        if c == "gb":
            cache = ROOT / "data" / "interim" / "orcid_uk_phd_faculty_edges.parquet"
            if cache.exists():
                put(sec, "GB house filter: rows in the scripts/40 cache (check)", int(len(pd.read_parquet(cache, columns=["person_orcid"]))))
        describe(sec, f"{c.upper()} + Italian titles (DOC|DOC_IT & FAC|FAC_IT)", x[x.doc_it & x.fac_it], uni)
        xb = x[x.doc_it & (x.fac_it | x.res) & x.org_to_ror_id.isin(uni)]
        describe(sec, f"{c.upper()} broad upper bound (+ 'researcher'/'ricercatore', destination ROR type education)", xb, uni)


# ----------------------------------------------------------------------------------------------
# (a) AlmaLaurea
# ----------------------------------------------------------------------------------------------
def select_options(s, name):
    m = re.search(r'<select name="%s"[^>]*>(.*?)</select>' % name, s, re.S)
    return re.findall(r"<option[^>]*value=['\"]([^'\"]*)['\"][^>]*>(.*?)</option>", m.group(1), re.S) if m else []


def section_a():
    sec = "a"
    d = RAW / "almalaurea"
    for f in sorted(d.glob("solotendine_occupazione_*.html")):
        y = int(re.search(r"(\d{4})", f.name).group(1))
        s = f.read_text(encoding="utf-8", errors="replace")
        at = [html.unescape(t) for v, t in select_options(s, "ateneo") if v != "tutti"]
        put(sec, f"AlmaLaurea survey {y}: universities selectable in the query tool", len(at))
        absent = [k for k in ["Bocconi", "Milano Politecnico", "Cattolica", "LUISS", "Luiss"] if not any(k in a for a in at)]
        put(sec, f"AlmaLaurea survey {y}: of Bocconi / Milano Politecnico / Cattolica / LUISS, not in the list",
            " / ".join(absent) if absent else "none")
        if y == 2025:
            put(sec, "survey years offered in the query tool", len(select_options(s, "anno")))
            put(sec, "years-since-graduation options (2025 survey)", " / ".join(t for v, t in select_options(s, "annolau")))
            put(sec, "course types (2025 survey)", " / ".join(t for v, t in select_options(s, "corstipo") if v != "tutti"))
            put(sec, "Facolta/Dipartimento/Scuola options (2025 survey, all universities)",
                len([v for v, t in select_options(s, "facolta") if v != "tutti"]))

    s = (d / "gli_atenei.html").read_text(encoding="utf-8", errors="replace")
    txt = html.unescape(re.sub(r"<[^>]+>", "\n", re.sub(r"<script.*?</script>", "", s, flags=re.S)))
    for k in ["Bocconi", "Politecnico di Milano", "Cattolica", "LUISS", "Luiss"]:
        put(sec, f"member list page mentions '{k}'", int(k in txt))

    from pypdf import PdfReader
    t = "\n".join((p.extract_text() or "") for p in PdfReader(str(d / "note_metodologiche_occupazione_2025.pdf")).pages)
    t1 = re.sub(r"\s+", " ", t)
    m = re.search(r"di (\d+) università italiane, delle (\d+) aderenti", t1)
    put(sec, "methodology notes 2025: universities surveyed / AlmaLaurea members (as stated)",
        f"{m.group(1)} / {m.group(2)}" if m else "not found")
    m = re.search(r"inferiore a (\d+) unità", t1)
    put(sec, "methodology notes 2025: statistics suppressed ('*') below this many graduates (as stated)",
        m.group(1) if m else "not found")

    s = (d / "visualizza_2025_LS_ateneo70003_annolau5_byclasse.html").read_text(encoding="utf-8", errors="replace")
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", s, re.S)
    lab_prev, n_cols, n_val = None, None, None
    for rr in rows:
        th = re.findall(r"<th[^>]*>(.*?)</th>", rr, re.S)
        lab = html.unescape(re.sub(r"<[^>]+>", "", re.sub(r"<span.*?</span>", "", th[0], flags=re.S))).strip() if th else ""
        vals = [html.unescape(re.sub(r"<[^>]+>", "", x)).replace("\xa0", "").strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", rr, re.S)]
        if lab.startswith("Retribuzione mensile netta"):
            lab_prev = "ret"
        elif lab == "Totale" and lab_prev == "ret":
            n_cols = len(vals)
            n_val = sum(1 for v in vals if re.fullmatch(r"[\d.]+", v))
            lab_prev = None
    put(sec, "sample result page (Bologna, LM biennale, 5 years, by class): columns (collective + classes)", n_cols)
    put(sec, "sample result page: columns with a net-monthly-earnings value", n_val)
    tip = re.search(r"La domanda relativa alla retribuzione mensile netta prevede fasce di (\d+) euro", html.unescape(s))
    put(sec, "earnings question: band width in euro (tooltip on the result page)", tip.group(1) if tip else "not found")
    tip_full = re.search(r"(La domanda relativa alla retribuzione mensile netta.*?)Per ulteriori", html.unescape(s), re.S)
    put(sec, "earnings tooltip, verbatim", re.sub(r"\s+", " ", tip_full.group(1).replace("\\'", "'")).strip()
        if tip_full else "not found")
    put(sec, "result page offers a CSV export", int("Esportazione dati per foglio elettronico" in html.unescape(s)))


# ----------------------------------------------------------------------------------------------
# (c) MUR / USTAT catalogue; medicine test
# ----------------------------------------------------------------------------------------------
def section_c():
    sec = "c"
    j = json.loads((RAW / "catalog" / "dati_gov_it_mur_packages.json").read_text(encoding="utf-8"))["result"]
    put(sec, "MUR datasets in the national catalogue (dati.gov.it, organisation ministero-universita-e-ricerca)", j["count"])
    lic = sorted({p.get("license_title") for p in j["results"]})
    put(sec, "licences of those datasets", " | ".join(str(x) for x in lic))
    want = {"iscritti", "laureati", "immatricolati-nuovi-ingressi-e-iscritti-al-1-anno", "metadati", "formazione-post-laurea"}
    for p in j["results"]:
        if p["name"] in want:
            res = p["resources"]
            put(sec, f"dataset '{p['name']}': resources", len(res))
            for r in res:
                u = r.get("url", "")
                n = r.get("name", "")
                if re.search(r"corso|voto|offerta|classe|atene", n, re.I) and not re.search(r"internazional|master|specializz|esami", n, re.I):
                    put(sec, f"  {p['name']}: {n.strip()}", u.split("/download/")[-1])

    d = RAW / "medicina_anon"
    for y in (2015, 2016):
        s = (d / f"ME_RI_{y}.html").read_text(encoding="utf-8", errors="replace")
        put(sec, f"medicine test {y}: anonymous-result PDFs linked (one per test-site university)",
            len(set(re.findall(r"/%d/risultati/MED/[^\"']+\.pdf" % y, s))))
    from pypdf import PdfReader
    rd = PdfReader(str(d / "2016_MED_01.pdf"))
    t = "\n".join((p.extract_text() or "") for p in rd.pages)
    hdr = re.search(r"Codice[^\n]*", t)
    put(sec, "sample PDF (2016, site 01): pages", len(rd.pages))
    put(sec, "sample PDF: column header", hdr.group(0).strip() if hdr else "not found")
    put(sec, "sample PDF: candidate rows (anonymous label code)", len(re.findall(r"^\w{15}\b", t, re.M)))
    put(sec, "sample PDF: of which with label code + 6 scores",
        len(re.findall(r"^\w{15}(?:\s+-?[\d.]+){6}\s*$", t, re.M)))
    put(sec, "sample PDF: of which flagged 'Compito Identificabile' (no scores)",
        len(re.findall(r"^\w{15}\s+Compito Identificabile", t, re.M)))
    s = (d / "MED_2023.html").read_text(encoding="utf-8", errors="replace")
    put(sec, "2023/24 page: national ranking behind the reserved area",
        int("area riservata" in html.unescape(s) and "ap-graduatorie.cineca.it" in s))


def main():
    section_a()
    section_b()
    section_c()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ROWS).to_csv(OUT, index=False)
    print(f"wrote {OUT.relative_to(ROOT)} ({len(ROWS)} rows)")


if __name__ == "__main__":
    main()
