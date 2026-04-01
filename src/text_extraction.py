import os
import re
import json
import pymupdf
import pytesseract
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple

# ── Paths ──────────────────────────────────────────────────────────────────────
source_data         = os.path.join(os.path.dirname(__file__), "..", "Source_Data")
sample_report_path  = os.path.join(source_data, "Sample Report.pdf")
thermal_images_path = os.path.join(source_data, "Thermal Images.pdf")
images_dir          = os.path.join(os.path.dirname(__file__), "..", "extracted_images")
output_dir          = os.path.join(os.path.dirname(__file__), "..", "output")


# ── Data Classes ───────────────────────────────────────────────────────────────
@dataclass
class AreaInfo:
    area_number: int
    negative_side: str = "N/A"
    positive_side: str = "N/A"
    negative_side_photos: List[str] = field(default_factory=list)
    positive_side_photos: List[str] = field(default_factory=list)

    def to_dict(self):
        return {
            "area_number": self.area_number,
            "negative_side_description": self.negative_side,
            "positive_side_description": self.positive_side,
            "negative_side_photos": self.negative_side_photos,
            "positive_side_photos": self.positive_side_photos,
        }


@dataclass
class SummaryRow:
    point_no: str
    impacted_area: str
    ref_point_no: str
    exposed_area: str

    def to_dict(self):
        return {
            "point_no": self.point_no,
            "impacted_area_negative_side": self.impacted_area,
            "ref_point_no": self.ref_point_no,
            "exposed_area_positive_side": self.exposed_area,
        }


@dataclass
class ChecklistItem:
    question: str
    answer: str

    def to_dict(self):
        return {"question": self.question, "answer": self.answer}


@dataclass
class ChecklistSection:
    name: str
    score: str = "N/A"
    items: List[ChecklistItem] = field(default_factory=list)

    def to_dict(self):
        return {"name": self.name, "score": self.score, "items": [i.to_dict() for i in self.items]}


@dataclass
class SiteInfo:
    impacted_areas: List[str] = field(default_factory=list)
    areas: List[AreaInfo] = field(default_factory=list)
    summary_table: List[SummaryRow] = field(default_factory=list)
    checklists: List[ChecklistSection] = field(default_factory=list)
    appendix_photos: List[str] = field(default_factory=list)

    def to_json(self):
        return {
            "impacted_areas": self.impacted_areas,
            "areas": [a.to_dict() for a in self.areas],
            "summary_table": [r.to_dict() for r in self.summary_table],
            "checklists": [c.to_dict() for c in self.checklists],
            "appendix_photos": self.appendix_photos,
        }


@dataclass
class ThermalReading:
    page: int
    image_filename: str
    date: str
    hotspot: str
    coldspot: str
    emissivity: str
    reflected_temp: str
    thermal_scan_path: Optional[str] = None
    photo_path: Optional[str] = None

    def to_dict(self):
        return {
            "page": self.page,
            "image_filename": self.image_filename,
            "date": self.date,
            "hotspot": self.hotspot,
            "coldspot": self.coldspot,
            "emissivity": self.emissivity,
            "reflected_temperature": self.reflected_temp,
            "thermal_scan_path": self.thermal_scan_path,
            "photo_path": self.photo_path,
        }


# ── Low-level helpers ──────────────────────────────────────────────────────────
def _get_page_text(page: pymupdf.Page) -> str:
    text = page.get_text().strip()
    if text:
        return text
    pix   = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    nparr = np.frombuffer(pix.tobytes("ppm"), np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.fastNlMeansDenoising(gray, h=10)
    return pytesseract.image_to_string(gray)


def _pv(pattern, text, default="N/A"):
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else default


def _clean(s: str) -> str:
    return " ".join(s.split()).strip()


def _save_img(doc: pymupdf.Document, xref: int, path: str) -> str:
    pix = pymupdf.Pixmap(doc, xref)
    if pix.colorspace and pix.colorspace.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pix.save(path)
    return os.path.normpath(path)


def _page_text_positions(page: pymupdf.Page) -> List[Tuple[float, str]]:
    """Return [(y, text)] for every text line, sorted top→bottom."""
    result = []
    for block in page.get_text("dict").get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            txt = " ".join(s["text"] for s in line.get("spans", []))
            if txt.strip():
                result.append((line["bbox"][1], txt.strip()))
    return sorted(result, key=lambda x: x[0])


def _large_image_rects(page: pymupdf.Page, min_area=2500):
    """Return [(rect, xref)] for non-tiny images, sorted top→bottom."""
    seen, out = set(), []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            for r in page.get_image_rects(xref):
                if r.width * r.height >= min_area:
                    key = (round(r.x0, 1), round(r.y0, 1))
                    if key not in seen:
                        seen.add(key)
                        out.append((r, xref))
        except Exception:
            pass
    out.sort(key=lambda x: x[0].y0)
    return out


def _classify_page_images(page: pymupdf.Page, fallback: Optional[str]) -> Dict[int, str]:
    """
    Map xref → 'negative'|'positive' using photo-section heading y-positions.
    Falls back to `fallback` when no heading is found on this page.
    """
    neg_y = pos_y = None
    for y, txt in _page_text_positions(page):
        if neg_y is None and re.search(r"Negative\s+side\s+photo", txt, re.IGNORECASE):
            neg_y = y
        if pos_y is None and re.search(r"Positive\s+side\s+photo", txt, re.IGNORECASE):
            pos_y = y

    result = {}
    for img in page.get_images(full=True):
        xref = img[0]
        side = fallback
        try:
            rects = page.get_image_rects(xref)
            if rects:
                iy = rects[0].y0
                if neg_y is not None and pos_y is not None:
                    side = "negative" if iy < pos_y else "positive"
                elif neg_y is not None:
                    side = "negative"
                elif pos_y is not None:
                    side = "positive"
        except Exception:
            pass
        if side:
            result[xref] = side
    return result


# ── Text-level parsers (operate on full_text) ──────────────────────────────────
def _section_text(full_text: str, start_re: str, end_patterns: List[str]) -> str:
    m = re.search(start_re, full_text, re.IGNORECASE)
    if not m:
        return ""
    chunk = full_text[m.start():]
    for pat in end_patterns:
        e = re.search(pat, chunk, re.IGNORECASE)
        if e:
            chunk = chunk[:e.start()]
    return chunk


def _parse_impacted_areas(full_text: str) -> List[str]:
    """Extract rooms list from 'Impacted Areas/Rooms' — single-line, comma-split."""
    m = re.search(r"Impacted\s+Areas?[/\s]*Rooms?\s*\n?(.*?)(?:\n\n|\nImpacted\s+Area\b|\Z)",
                  full_text, re.IGNORECASE | re.DOTALL)
    if not m:
        return []
    raw = m.group(1).strip()
    tokens = [_clean(t) for t in re.split(r",|\n", raw) if _clean(t)]
    noise  = re.compile(r"^(impacted\s+area|negative|positive|photo\s+\d+|site\s+details|\d+)$", re.IGNORECASE)
    return [t for t in tokens if not noise.match(t)]


def _parse_area_descriptions(full_text: str) -> Dict[int, AreaInfo]:
    """Extract all numbered area blocks and their neg/pos descriptions."""
    area_map: Dict[int, AreaInfo] = {}
    # Grab section from first numbered area to checklists
    section = _section_text(full_text,
                            r"Impacted\s+Area\s+\d+",
                            [r"Inspection\s+Checklists?", r"SUMMARY\s+TABLE"])
    starts = list(re.finditer(r"Impacted\s+Area\s+(\d+)\b", section, re.IGNORECASE))
    for i, m in enumerate(starts):
        num   = int(m.group(1))
        start = m.start()
        end   = starts[i + 1].start() if i + 1 < len(starts) else len(section)
        block = section[start:end]

        area = AreaInfo(area_number=num)
        neg = re.search(r"Negative\s+side\s+Description\s+(.*?)(?=\nNegative\s+side\s+photo|\nPositive\s+side|\Z)",
                        block, re.IGNORECASE | re.DOTALL)
        if neg:
            area.negative_side = _clean(neg.group(1))

        pos = re.search(r"Positive\s+side\s+Description\s+(.*?)(?=\nPositive\s+side\s+photo|\nImpacted\s+Area|\Z)",
                        block, re.IGNORECASE | re.DOTALL)
        if pos:
            area.positive_side = _clean(pos.group(1))

        area_map[num] = area
    return area_map


def _parse_summary_table(full_text: str) -> List[SummaryRow]:
    rows    = []
    section = _section_text(full_text, r"SUMMARY\s+TABLE", [r"Appendix", r"Inspection\s+Checklist"])
    lines   = [l.strip() for l in section.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        if re.fullmatch(r"\d+", lines[i]):
            neg_pt = lines[i]; i += 1
            neg_parts = []
            while i < len(lines) and not re.fullmatch(r"\d+\.\d+|\d+", lines[i]):
                neg_parts.append(lines[i]); i += 1
            ref_pt = pos_parts = ""
            if i < len(lines) and re.fullmatch(r"\d+\.\d+", lines[i]):
                ref_pt = lines[i]; i += 1
                pos_acc = []
                while i < len(lines) and not re.fullmatch(r"\d+", lines[i]) and not re.fullmatch(r"\d+\.\d+", lines[i]):
                    pos_acc.append(lines[i]); i += 1
                pos_parts = " ".join(pos_acc)
            rows.append(SummaryRow(neg_pt, " ".join(neg_parts), ref_pt, pos_parts))
        else:
            i += 1
    return rows


def _parse_checklists(full_text: str) -> List[ChecklistSection]:
    section = _section_text(full_text, r"Inspection\s+Checklists?", [r"SUMMARY\s+TABLE"])
    if not section:
        return []

    ANSWERS  = re.compile(r"^(Yes|No|N/A|Moderate|All\s*time|Not\s*sure|\d+\.?\d*%|Vitrified|Ceramic|Good|Poor|Fair)$", re.IGNORECASE)
    SUBSEC   = re.compile(r"(Negative\s+Side\s+Inputs|Positive\s+Side\s+Inputs|Structural\s+Condition\s+of\s+RCC|Condition\s+of\s+External\s+wall|Condition\s+of\s+Adhesion)", re.IGNORECASE)
    NOISE    = re.compile(r"^(Inspection\s+Checklists?|Photo\s+\d+|Impacted\s+Area\s*\d*|Negative\s+side\s+(photo|desc)|Positive\s+side\s+(photo|desc)|Checklists?\s*:?|Appendix|\d+\s+flagged|SUMMARY)$", re.IGNORECASE)
    SCOREONLY= re.compile(r"^\d+\.?\d*%$")

    lines = [l.strip() for l in section.splitlines() if l.strip()]
    checklists: List[ChecklistSection] = []
    cur_name  = "General"
    cur_score = "N/A"
    cur_items : List[ChecklistItem] = []

    def flush():
        if cur_items:
            checklists.append(ChecklistSection(name=cur_name, score=cur_score, items=list(cur_items)))

    i = 0
    while i < len(lines):
        ln = lines[i]
        if NOISE.match(ln):               i += 1; continue
        if SCOREONLY.match(ln):           cur_score = ln; i += 1; continue
        if SUBSEC.search(ln):
            flush(); cur_name = ln; cur_score = "N/A"; cur_items = []; i += 1; continue

        # Try single-line Q + answer
        if i + 1 < len(lines) and ANSWERS.match(lines[i + 1]) and not NOISE.match(ln):
            cur_items.append(ChecklistItem(question=ln, answer=lines[i + 1])); i += 2; continue
        # Try two-line Q + answer
        if i + 2 < len(lines) and not ANSWERS.match(lines[i + 1]) and ANSWERS.match(lines[i + 2]) and not NOISE.match(ln):
            cur_items.append(ChecklistItem(question=ln + " " + lines[i + 1], answer=lines[i + 2])); i += 3; continue

        i += 1

    flush()
    return checklists


# ── Sample Report Extraction ───────────────────────────────────────────────────
def extract_site_info(pdf_path: str = sample_report_path) -> SiteInfo:
    doc      = pymupdf.open(pdf_path)
    site     = SiteInfo()
    full_text= "\n".join(_get_page_text(p) for p in doc)

    # Text-level extractions
    site.impacted_areas = _parse_impacted_areas(full_text)
    area_map            = _parse_area_descriptions(full_text)
    site.summary_table  = _parse_summary_table(full_text)
    site.checklists     = _parse_checklists(full_text)

    # Page-by-page image saving
    current_area: Optional[int] = None
    current_side: Optional[str] = None
    in_appendix                 = False
    appendix_count              = 0
    counters: Dict[str, int]    = {}   # "areaN_side" → image count

    for page_num, page in enumerate(doc):
        text = _get_page_text(page)

        # Appendix detection
        if re.search(r"^\s*Appendix\s*$", text, re.IGNORECASE | re.MULTILINE):
            in_appendix = True

        if in_appendix:
            for img in page.get_images(full=True):
                appendix_count += 1
                path = _save_img(doc, img[0], os.path.join(
                    images_dir, "sample_doc", "appendix", f"img_{appendix_count}.png"))
                site.appendix_photos.append(path)
            continue

        # Skip non-area pages (checklist/summary pages, no impacted area text)
        if re.search(r"Inspection\s+Checklists?|SUMMARY\s+TABLE", text, re.IGNORECASE) \
                and not re.search(r"Impacted\s+Area\s+\d+", text, re.IGNORECASE):
            continue

        # Update current area: use LAST area number found on this page
        all_areas = re.findall(r"Impacted\s+Area\s+(\d+)", text, re.IGNORECASE)
        if all_areas:
            current_area = int(all_areas[-1])
            if current_area not in area_map:
                area_map[current_area] = AreaInfo(area_number=current_area)

        if current_area is None:
            continue

        # Decide side per image using position-aware classification
        page_sides = _classify_page_images(page, fallback=current_side)

        # Update current_side from page headings for carry-forward
        for _, ln in _page_text_positions(page):
            if re.search(r"Negative\s+side\s+photo", ln, re.IGNORECASE):
                current_side = "negative"
            if re.search(r"Positive\s+side\s+photo", ln, re.IGNORECASE):
                current_side = "positive"

        for img in page.get_images(full=True):
            xref = img[0]
            side = page_sides.get(xref)
            if not side:
                continue
            key = f"{current_area}_{side}"
            counters[key] = counters.get(key, 0) + 1
            path = _save_img(doc, xref, os.path.join(
                images_dir, "sample_doc",
                f"impacted_area_{current_area}", side,
                f"img_{counters[key]}.png"))
            if side == "negative":
                area_map[current_area].negative_side_photos.append(path)
            else:
                area_map[current_area].positive_side_photos.append(path)

    doc.close()
    site.areas = sorted(area_map.values(), key=lambda a: a.area_number)
    return site


# ── Thermal Report Extraction ──────────────────────────────────────────────────
def extract_thermal_data(pdf_path: str = thermal_images_path) -> List[ThermalReading]:
    doc      = pymupdf.open(pdf_path)
    readings = []
    for page_num, page in enumerate(doc):
        text = _get_page_text(page)
        r = ThermalReading(
            page           = page_num + 1,
            image_filename = _pv(r"Thermal\s+image\s*[:\-]?\s*(\S+)", text),
            date           = _pv(r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", text),
            hotspot        = _pv(r"Hotspot\s*[:\-]?\s*([\d.]+\s*°?\s*C)", text),
            coldspot       = _pv(r"Coldspot\s*[:\-]?\s*([\d.]+\s*°?\s*C)", text),
            emissivity     = _pv(r"Emissivity\s*[:\-]?\s*([\d.]+)", text),
            reflected_temp = _pv(r"Reflected\s+temperature\s*[:\-]?\s*([\d.]+\s*°?\s*C)", text),
        )
        folder     = os.path.join(images_dir, "thermal_doc", f"page_{page_num + 1}")
        placements = _large_image_rects(page)
        scale      = pymupdf.Matrix(2, 2)
        if len(placements) >= 1:
            rect, _ = placements[0]
            pix = page.get_pixmap(matrix=scale, clip=rect)
            path = os.path.join(folder, "thermal_scan.png")
            os.makedirs(folder, exist_ok=True); pix.save(path)
            r.thermal_scan_path = os.path.normpath(path)
        if len(placements) >= 2:
            rect, _ = placements[1]
            pix = page.get_pixmap(matrix=scale, clip=rect)
            path = os.path.join(folder, "photo.png")
            pix.save(path)
            r.photo_path = os.path.normpath(path)
        readings.append(r)
    doc.close()
    return readings


# ── Combined Export ────────────────────────────────────────────────────────────
def extract_all_to_json(save: bool = True) -> dict:
    print("► Extracting Sample Report...")
    site = extract_site_info()
    print("► Extracting Thermal Images...")
    readings = extract_thermal_data()
    result = {
        "sample_report": site.to_json(),
        "thermal_report": [r.to_dict() for r in readings],
    }
    if save:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "extracted_data.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"✓ JSON → {path}")
        print(f"✓ Images → {images_dir}")
    return result


if __name__ == "__main__":
    data = extract_all_to_json(save=True)
    sr   = data["sample_report"]
    print(f"\n=== SAMPLE REPORT ===")
    print(f"  Rooms          : {', '.join(sr['impacted_areas'])}")
    print(f"  Areas found    : {len(sr['areas'])}")
    print(f"  Summary rows   : {len(sr['summary_table'])}")
    print(f"  Checklist secs : {len(sr['checklists'])}")
    print(f"  Appendix imgs  : {len(sr['appendix_photos'])}")
    print(f"\n=== THERMAL ===")
    print(f"  Pages : {len(data['thermal_report'])}")
