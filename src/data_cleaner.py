import os
import re
import json


# ── Paths ──────────────────────────────────────────────────────────────────────
_BASE        = os.path.join(os.path.dirname(__file__), "..")
INPUT_JSON   = os.path.normpath(os.path.join(_BASE, "output", "extracted_data.json"))
OUTPUT_JSON  = os.path.normpath(os.path.join(_BASE, "output", "cleaned_data.json"))
IMAGES_DIR   = os.path.normpath(os.path.join(_BASE, "extracted_images"))


# ── Helpers ────────────────────────────────────────────────────────────────────
def _norm_temp(s: str) -> str:
    """Normalise temperature string → '28.8°C'."""
    if not s or s == "N/A":
        return s
    s = re.sub(r"\s+", "", s)                    # remove internal spaces
    s = re.sub(r"°?C$", "°C", s, flags=re.IGNORECASE)
    # ensure ° is present
    if s.endswith("C") and "°" not in s:
        s = s[:-1] + "°C"
    return s


def _norm_path(p: str) -> str:
    """Resolve '..' and normalise separators."""
    return os.path.normpath(p) if p else p


def _path_exists(p: str) -> bool:
    return bool(p) and os.path.isfile(p)


def _dedup(lst: list) -> list:
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


_NOISE_DESC = re.compile(
    r"^(N/A|None|—|-+|n/a|na)$", re.IGNORECASE
)

_NOISE_CHECKLIST_Q = re.compile(
    r"^(Photo\s+\d+|Impacted\s+Area\s*\d*|Negative\s+side|Positive\s+side"
    r"|Inspection\s+Checklists?|Appendix|SUMMARY\s+TABLE"
    r"|Checklists?\s*:?|\d+\s+flagged|\d+\.?\d*%"
    r"|WC|External\s+wall)$",
    re.IGNORECASE
)


# ── Section cleaners ──────────────────────────────────────────────────────────
def _clean_impacted_areas(areas: list) -> list:
    """Remove duplicates and structural noise from room names."""
    cleaned = []
    seen    = set()
    for a in areas:
        a = " ".join(a.split())
        if not a or _NOISE_DESC.match(a):
            continue
        lo = a.lower()
        if lo not in seen:
            seen.add(lo)
            cleaned.append(a)
    return cleaned


def _clean_areas(areas: list) -> list:
    for area in areas:
        # Join description fragments if they look truncated (no sentence-ending char)
        for key in ("negative_side_description", "positive_side_description"):
            val = area.get(key, "N/A")
            if _NOISE_DESC.match(str(val)):
                area[key] = "N/A"
            else:
                area[key] = " ".join(val.split())   # collapse whitespace

        # Normalise + deduplicate photo lists
        for key in ("negative_side_photos", "positive_side_photos"):
            raw   = area.get(key, [])
            paths = [_norm_path(p) for p in raw]
            area[key] = _dedup(paths)

        # Add photo count metadata
        area["negative_photo_count"] = len(area.get("negative_side_photos", []))
        area["positive_photo_count"] = len(area.get("positive_side_photos", []))

    return areas


def _clean_summary_table(rows: list) -> list:
    cleaned = []
    for row in rows:
        # Skip header-like rows that slipped through
        if row.get("point_no", "").lower() in ("point", "no", ""):
            continue
        for key in ("impacted_area_negative_side", "exposed_area_positive_side"):
            row[key] = " ".join(row.get(key, "").split())
        cleaned.append(row)
    return cleaned


def _clean_checklists(sections: list) -> list:
    cleaned_sections = []
    for sec in sections:
        items = []
        for item in sec.get("items", []):
            q = " ".join(item.get("question", "").split())
            a = " ".join(item.get("answer", "").split())
            if _NOISE_CHECKLIST_Q.match(q):
                continue
            if not q or not a:
                continue
            items.append({"question": q, "answer": a})
        if items:
            cleaned_sections.append({
                "name" : sec.get("name", "Unknown"),
                "score": sec.get("score", "N/A"),
                "items": items,
            })
    return cleaned_sections


def _clean_thermal(readings: list) -> list:
    for r in readings:
        for key in ("hotspot", "coldspot", "reflected_temperature"):
            r[key] = _norm_temp(r.get(key, "N/A"))
        for key in ("thermal_scan_path", "photo_path"):
            r[key] = _norm_path(r.get(key, "")) or None
    return readings


def _validate_and_report(data: dict):
    """Print a quick validation summary."""
    sr     = data["sample_report"]
    tr     = data["thermal_report"]
    errors = []

    if not sr.get("impacted_areas"):
        errors.append("sample_report.impacted_areas is empty")

    for area in sr.get("areas", []):
        n = area["area_number"]
        if area["negative_side_description"] == "N/A":
            errors.append(f"  Area {n}: negative_side_description missing")
        if area["positive_side_description"] == "N/A":
            errors.append(f"  Area {n}: positive_side_description missing")
        total = area["negative_photo_count"] + area["positive_photo_count"]
        if total == 0:
            errors.append(f"  Area {n}: no photos found")

    for r in tr:
        if not r.get("thermal_scan_path"):
            errors.append(f"  Thermal page {r['page']}: no thermal_scan_path")

    if errors:
        print("\n⚠  Validation issues:")
        for e in errors:
            print(f"   {e}")
    else:
        print("\n✓ Validation passed — no issues found")


# ── Main ───────────────────────────────────────────────────────────────────────
def clean(input_path: str = INPUT_JSON, output_path: str = OUTPUT_JSON) -> dict:
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    sr = data["sample_report"]
    sr["impacted_areas"] = _clean_impacted_areas(sr.get("impacted_areas", []))
    sr["areas"]          = _clean_areas(sr.get("areas", []))
    sr["summary_table"]  = _clean_summary_table(sr.get("summary_table", []))
    sr["checklists"]     = _clean_checklists(sr.get("checklists", []))
    sr["appendix_photos"]= _dedup([_norm_path(p) for p in sr.get("appendix_photos", [])])

    data["thermal_report"] = _clean_thermal(data.get("thermal_report", []))
    data["sample_report"]  = sr

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    _validate_and_report(data)
    print(f"✓ Cleaned JSON → {output_path}")
    return data


if __name__ == "__main__":
    result = clean()
    sr = result["sample_report"]
    print(f"\n=== CLEANED SUMMARY ===")
    print(f"  Impacted areas  : {sr['impacted_areas']}")
    print(f"  Areas extracted : {len(sr['areas'])}")
    for a in sr["areas"]:
        print(f"    Area {a['area_number']}: neg_photos={a['negative_photo_count']}  pos_photos={a['positive_photo_count']}")
    print(f"  Summary rows    : {len(sr['summary_table'])}")
    print(f"  Checklist secs  : {len(sr['checklists'])}")
    print(f"  Appendix photos : {len(sr['appendix_photos'])}")
    print(f"  Thermal pages   : {len(result['thermal_report'])}")
