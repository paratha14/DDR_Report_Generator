import os
import re
import json
import pymupdf
import pytesseract
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict

# ── PDF Paths ──────────────────────────────────────────────────────────────────
source_data         = os.path.join(os.path.dirname(__file__), "..", "Source_Data")
sample_report_path  = os.path.join(source_data, "Sample Report.pdf")
thermal_images_path = os.path.join(source_data, "Thermal Images.pdf")

# ── Output directories ─────────────────────────────────────────────────────────
images_dir = os.path.join(os.path.dirname(__file__), "..", "extracted_images")
output_dir = os.path.join(os.path.dirname(__file__), "..", "output")


# ── Data Classes ───────────────────────────────────────────────────────────────
@dataclass
class AreaInfo:
    area_number: int
    negative_side: str = "N/A"
    positive_side: str = "N/A"
    negative_side_photos: List[str] = field(default_factory=list)
    positive_side_photos: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
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
    impacted_area: str          # negative side description
    ref_point_no: str
    exposed_area: str           # positive side description

    def to_dict(self) -> dict:
        return {
            "point_no": self.point_no,
            "impacted_area_negative_side": self.impacted_area,
            "ref_point_no": self.ref_point_no,
            "exposed_area_positive_side": self.exposed_area,
        }


@dataclass
class ChecklistSection:
    name: str                           # e.g. "WC", "External wall"
    score: str = "N/A"                  # e.g. "84.21%"
    flagged_count: str = "N/A"
    items: List[Dict[str, str]] = field(default_factory=list)  # [{"item": ..., "value": ...}]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "score": self.score,
            "flagged_count": self.flagged_count,
            "items": self.items,
        }


@dataclass
class SiteInfo:
    impacted_areas: List[str] = field(default_factory=list)     # rooms/zones (site location)
    areas: List[AreaInfo] = field(default_factory=list)          # impacted area details
    summary_table: List[SummaryRow] = field(default_factory=list)
    checklists: List[ChecklistSection] = field(default_factory=list)
    appendix_photos: List[str] = field(default_factory=list)     # paths to 64 appendix images

    def to_json(self) -> dict:
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

    def to_dict(self) -> dict:
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _get_page_text(page: pymupdf.Page) -> str:
    """Return native text; fall back to Tesseract OCR for scanned pages."""
    text = page.get_text().strip()
    if text:
        return text
    pix   = page.get_pixmap(matrix=pymupdf.Matrix(2, 2))
    nparr = np.frombuffer(pix.tobytes("ppm"), np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray  = cv2.fastNlMeansDenoising(gray, h=10)
    return pytesseract.image_to_string(gray)


def _parse_value(pattern: str, text: str, default: str = "N/A") -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    return m.group(1).strip() if m else default


def _save_image(doc: pymupdf.Document, xref: int, save_path: str) -> str:
    """Extract an embedded image by xref and save as PNG."""
    pix = pymupdf.Pixmap(doc, xref)
    if pix.colorspace and pix.colorspace.n > 3:
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pix.save(save_path)
    return save_path


def _get_image_rects(page: pymupdf.Page, min_area: float = 2500):
    """
    Return a list of (rect, xref) for all non-tiny images on the page,
    sorted top-to-bottom by their y-position.
    Icons (hotspot/coldspot crosshairs, emissivity symbol, etc.) have a
    small bounding box on the page and are filtered out by min_area.
    """
    placements = []
    seen = set()
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            rects = page.get_image_rects(xref)
            for r in rects:
                area = r.width * r.height
                if area >= min_area:
                    key = (round(r.x0, 1), round(r.y0, 1))
                    if key not in seen:
                        seen.add(key)
                        placements.append((r, xref))
        except Exception:
            pass
    placements.sort(key=lambda x: x[0].y0)   # top → bottom
    return placements


# ── Summary Table Parser ───────────────────────────────────────────────────────
def _parse_summary_table(text: str) -> List[SummaryRow]:
    """
    Parse the SUMMARY TABLE section.
    Each row: point_no | impacted area description | ref_point_no | exposed area description
    """
    rows = []
    section_match = re.search(
        r"SUMMARY\s+TABLE(.+?)(?:Appendix|Inspection\s+Checklist|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if not section_match:
        return rows

    section = section_match.group(1)
    lines = [l.strip() for l in section.splitlines() if l.strip()]

    # Lines look like: number, description, decimal_number, description (interleaved)
    # Group by detecting lines that are purely point numbers (e.g. "1", "2", "1.1", "2.1")
    i = 0
    neg_point = neg_desc = ref_point = pos_desc = ""
    while i < len(lines):
        line = lines[i]
        # Main point number (integer)
        if re.fullmatch(r"\d+", line):
            neg_point = line
            neg_desc  = lines[i + 1] if i + 1 < len(lines) else ""
            # Concatenate continuation lines until we hit a decimal point number or end
            j = i + 2
            while j < len(lines) and not re.fullmatch(r"\d+\.\d+", lines[j]):
                neg_desc += " " + lines[j]
                j += 1
            # Decimal reference point (e.g. 1.1)
            if j < len(lines) and re.fullmatch(r"\d+\.\d+", lines[j]):
                ref_point = lines[j]
                pos_desc  = lines[j + 1] if j + 1 < len(lines) else ""
                k = j + 2
                while k < len(lines) and not re.fullmatch(r"\d+", lines[k]) and not re.fullmatch(r"\d+\.\d+", lines[k]):
                    pos_desc += " " + lines[k]
                    k += 1
                i = k
            else:
                i = j
            rows.append(SummaryRow(
                point_no     = neg_point,
                impacted_area= neg_desc.strip(),
                ref_point_no = ref_point,
                exposed_area = pos_desc.strip(),
            ))
        else:
            i += 1

    return rows


# ── Checklist Parser ───────────────────────────────────────────────────────────
def _parse_checklists(text: str) -> List[ChecklistSection]:
    """
    Parse the Inspection Checklists section.
    Each sub-section has a name, score, flagged count, and key-value items.
    """
    checklists = []
    section_match = re.search(
        r"Inspection\s+Checklists?(.+?)(?:SUMMARY\s+TABLE|\Z)",
        text, re.IGNORECASE | re.DOTALL
    )
    if not section_match:
        return checklists

    section = section_match.group(1)

    # Sub-sections are headed by checklist names like "WC", "External wall", etc.
    # preceded by a score percentage and "N flagged"
    sub_sections = re.split(r"(?=Checklist\s*:)", section, flags=re.IGNORECASE)

    for sub in sub_sections:
        if not sub.strip():
            continue
        lines = [l.strip() for l in sub.splitlines() if l.strip()]
        if not lines:
            continue

        # Score and flagged count appear before the checklist items
        score_match   = re.search(r"(\d+\.?\d*%)", sub)
        flagged_match = re.search(r"(\d+)\s+flagged", sub, re.IGNORECASE)
        name_match    = re.search(r"Checklist\s*:\s*(.+?)(?:\n|$)", sub, re.IGNORECASE)

        name    = name_match.group(1).strip() if name_match else lines[0]
        score   = score_match.group(1) if score_match else "N/A"
        flagged = flagged_match.group(1) + " flagged" if flagged_match else "N/A"

        # Key-value items: question line followed by answer (Yes/No/N/A/Moderate/percentage/text)
        items = []
        answer_keywords = re.compile(
            r"^(Yes|No|N/A|Moderate|All\s+time|Not\s+sure|\d+\.?\d*%|[\w\s]+)$",
            re.IGNORECASE
        )
        # Extract after "Checklist:" header block
        item_section = re.sub(r".*?Checklist\s*:.*?\n", "", sub, count=1, flags=re.IGNORECASE | re.DOTALL)
        item_lines = [l.strip() for l in item_section.splitlines() if l.strip()]

        i = 0
        while i < len(item_lines):
            q = item_lines[i]
            # Skip pure percentage/score lines at top level
            if re.fullmatch(r"\d+\.?\d*%", q) or re.fullmatch(r"\d+\s+flagged", q, re.IGNORECASE):
                i += 1
                continue
            # Next non-empty line is the answer if it's short
            if i + 1 < len(item_lines):
                a = item_lines[i + 1]
                if len(a) <= 30 and answer_keywords.match(a):
                    items.append({"item": q, "value": a})
                    i += 2
                    continue
            items.append({"item": q, "value": "—"})
            i += 1

        checklists.append(ChecklistSection(
            name=name, score=score, flagged_count=flagged, items=items
        ))

    return checklists


# ── Sample Report Extraction ───────────────────────────────────────────────────

# Section states for the page-by-page pass
_SEC_AREA      = "impacted_area"
_SEC_CHECKLIST = "checklist"
_SEC_SUMMARY   = "summary_table"
_SEC_APPENDIX  = "appendix"


def extract_site_info(pdf_path: str = sample_report_path) -> SiteInfo:
    """
    Extracts from Sample Report.pdf:
      - impacted_areas     → physical location (rooms/zones)
      - areas              → per-area descriptions + photos
      - summary_table      → structured summary rows
      - checklists         → checklist items and scores
      - appendix_photos    → 64 appendix images saved to disk
    """
    doc  = pymupdf.open(pdf_path)
    site = SiteInfo()
    full_text = "\n".join(_get_page_text(p) for p in doc)

    # ── 1. Site location: only the rooms line ─────────────────────────────────
    rooms_match = re.search(
        r"Impacted\s+Areas?[/\s]*Rooms?\s*[:\-]?\s*(.+?)(?:\n)",
        full_text, re.IGNORECASE
    )
    if rooms_match:
        raw = rooms_match.group(1).strip()
        site.impacted_areas = [r.strip() for r in re.split(r",", raw) if r.strip()]

    # ── 2. Summary table ──────────────────────────────────────────────────────
    site.summary_table = _parse_summary_table(full_text)

    # ── 3. Checklists ─────────────────────────────────────────────────────────
    site.checklists = _parse_checklists(full_text)

    # ── 4. Page-by-page: track sections → save images with structured labels ──
    current_section  : str          = _SEC_AREA
    current_area_num : Optional[int]= None
    current_side     : Optional[str]= None   # "negative" | "positive"
    area_map : Dict[int, AreaInfo]  = {}
    img_counter: Dict[str, int]     = {}
    appendix_count: int             = 0

    for page_num, page in enumerate(doc):
        text = _get_page_text(page)

        # ── Detect section transitions ─────────────────────────────────────────
        if re.search(r"^\s*Appendix\s*$", text, re.IGNORECASE | re.MULTILINE):
            current_section = _SEC_APPENDIX
        elif re.search(r"SUMMARY\s+TABLE", text, re.IGNORECASE):
            current_section = _SEC_SUMMARY
        elif re.search(r"Inspection\s+Checklists?", text, re.IGNORECASE):
            current_section = _SEC_CHECKLIST
        elif re.search(r"Impacted\s+Area\s+\d+", text, re.IGNORECASE):
            current_section = _SEC_AREA

        # ── Impacted area section ──────────────────────────────────────────────
        if current_section == _SEC_AREA:
            area_match = re.search(r"Impacted\s+Area\s+(\d+)", text, re.IGNORECASE)
            if area_match:
                current_area_num = int(area_match.group(1))
                if current_area_num not in area_map:
                    area_map[current_area_num] = AreaInfo(area_number=current_area_num)

            if current_area_num and re.search(r"Negative\s+side\s+Description", text, re.IGNORECASE):
                current_side = "negative"
                area_map[current_area_num].negative_side = _parse_value(
                    r"Negative\s+side\s+Description\s*[:\-]?\s*(.+?)(?:\n|Positive|$)", text
                )

            if current_area_num and re.search(r"Positive\s+side\s+Description", text, re.IGNORECASE):
                current_side = "positive"
                area_map[current_area_num].positive_side = _parse_value(
                    r"Positive\s+side\s+Description\s*[:\-]?\s*(.+?)(?:\n|$)", text
                )

            for img_info in page.get_images(full=True):
                if current_area_num is None or current_side is None:
                    continue
                xref = img_info[0]
                key  = f"area_{current_area_num}_{current_side}"
                img_counter[key] = img_counter.get(key, 0) + 1
                save_path = os.path.join(
                    images_dir, "sample_doc",
                    f"impacted_area_{current_area_num}",
                    current_side,
                    f"img_{img_counter[key]}.png"
                )
                _save_image(doc, xref, save_path)
                if current_side == "negative":
                    area_map[current_area_num].negative_side_photos.append(save_path)
                else:
                    area_map[current_area_num].positive_side_photos.append(save_path)

        # ── Appendix section ───────────────────────────────────────────────────
        elif current_section == _SEC_APPENDIX:
            for img_info in page.get_images(full=True):
                appendix_count += 1
                xref = img_info[0]
                save_path = os.path.join(
                    images_dir, "sample_doc", "appendix",
                    f"img_{appendix_count}.png"
                )
                _save_image(doc, xref, save_path)
                site.appendix_photos.append(save_path)

    doc.close()
    site.areas = sorted(area_map.values(), key=lambda a: a.area_number)
    return site


# ── Thermal Report Extraction ──────────────────────────────────────────────────
def extract_thermal_data(pdf_path: str = thermal_images_path) -> List[ThermalReading]:
    """
    Per page extracts from Thermal Images.pdf:
      - Metadata: image filename, date, hotspot, coldspot, emissivity, reflected temp
      - Images: thermal scan (img 1) and regular photo (img 2) saved to disk
    """
    doc      = pymupdf.open(pdf_path)
    readings = []

    for page_num, page in enumerate(doc):
        text = _get_page_text(page)

        reading = ThermalReading(
            page           = page_num + 1,
            image_filename = _parse_value(r"Thermal\s+image\s*[:\-]?\s*(\S+)", text),
            date           = _parse_value(r"(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})", text),
            hotspot        = _parse_value(r"Hotspot\s*[:\-]?\s*([\d.]+\s*°?\s*C)", text),
            coldspot       = _parse_value(r"Coldspot\s*[:\-]?\s*([\d.]+\s*°?\s*C)", text),
            emissivity     = _parse_value(r"Emissivity\s*[:\-]?\s*([\d.]+)", text),
            reflected_temp = _parse_value(r"Reflected\s+temperature\s*[:\-]?\s*([\d.]+\s*°?\s*C)", text),
        )

        page_folder = os.path.join(images_dir, "thermal_doc", f"page_{page_num + 1}")
        os.makedirs(page_folder, exist_ok=True)

        # Use page-render approach: find image bounding boxes on the page
        # (sorted top→bottom) then render each region with get_pixmap(clip=rect).
        # This captures the RENDERED output (with the thermal colour gradient)
        # rather than the raw embedded bytes (which are grayscale IR data).
        placements = _get_image_rects(page)

        scale = pymupdf.Matrix(2, 2)   # 2× zoom for sharpness

        if len(placements) >= 1:
            rect, _ = placements[0]    # topmost large image = thermal scan
            pix  = page.get_pixmap(matrix=scale, clip=rect)
            path = os.path.join(page_folder, "thermal_scan.png")
            pix.save(path)
            reading.thermal_scan_path = path

        if len(placements) >= 2:
            rect, _ = placements[1]    # second image (below) = regular photo
            pix  = page.get_pixmap(matrix=scale, clip=rect)
            path = os.path.join(page_folder, "photo.png")
            pix.save(path)
            reading.photo_path = path

        readings.append(reading)

    doc.close()
    return readings


# ── Combined JSON Export ───────────────────────────────────────────────────────
def extract_all_to_json(save: bool = True) -> dict:
    """
    Run both extractions, save all images, and return a unified JSON dict.
    Writes JSON to output/extracted_data.json when save=True.
    """
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
        json_path = os.path.join(output_dir, "extracted_data.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"✓ JSON saved    → {json_path}")
        print(f"✓ Images saved  → {images_dir}")

    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = extract_all_to_json(save=True)

    sr = data["sample_report"]
    print(f"\n=== SAMPLE REPORT ===")
    print(f"  Location (rooms)   : {', '.join(sr['impacted_areas'])}")
    print(f"  Impacted areas     : {len(sr['areas'])}")
    print(f"  Summary table rows : {len(sr['summary_table'])}")
    print(f"  Checklist sections : {len(sr['checklists'])}")
    print(f"  Appendix photos    : {len(sr['appendix_photos'])}")

    print(f"\n=== THERMAL REPORT ===")
    print(f"  Total pages : {len(data['thermal_report'])}")
    for r in data["thermal_report"][:3]:
        print(f"  Page {r['page']}: {r['image_filename']} | {r['hotspot']} / {r['coldspot']}")
    print("  ...")
