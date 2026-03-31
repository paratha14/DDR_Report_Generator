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
class SiteInfo:
    impacted_areas: List[str] = field(default_factory=list)   # site location
    areas: List[AreaInfo] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "impacted_areas": self.impacted_areas,
            "areas": [a.to_dict() for a in self.areas],
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
    if pix.colorspace and pix.colorspace.n > 3:  # convert CMYK → RGB
        pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pix.save(save_path)
    return save_path


# ── Sample Report Extraction ───────────────────────────────────────────────────
def extract_site_info(pdf_path: str = sample_report_path) -> SiteInfo:
    """
    Extracts from Sample Report.pdf:
      - impacted_areas  → physical site location (rooms/zones)
      - per area: negative/positive side descriptions + all photos saved to disk
    """
    doc  = pymupdf.open(pdf_path)
    site = SiteInfo()

    # Get full text for global fields
    full_text = "\n".join(_get_page_text(p) for p in doc)

    # ── Site location (Impacted Areas / Rooms) ─────────────────────────────────
    rooms_match = re.search(
        r"Impacted\s+Areas?[/\s]*Rooms?\s*[:\-]?\s*(.+?)(?:\n{2,}|\Z)",
        full_text, re.IGNORECASE | re.DOTALL
    )
    if rooms_match:
        site.impacted_areas = [
            r.strip() for r in re.split(r"[,\n]+", rooms_match.group(1).strip()) if r.strip()
        ]

    # ── Page-by-page: track area / side → save images with structured labels ───
    current_area_num: Optional[int] = None
    current_side: Optional[str]     = None   # "negative" | "positive"
    area_map: Dict[int, AreaInfo]   = {}
    img_counter: Dict[str, int]     = {}     # "area_N_side" → count

    for page_num, page in enumerate(doc):
        text = _get_page_text(page)

        # Detect impacted area section
        area_match = re.search(r"Impacted\s+Area\s+(\d+)", text, re.IGNORECASE)
        if area_match:
            current_area_num = int(area_match.group(1))
            if current_area_num not in area_map:
                area_map[current_area_num] = AreaInfo(area_number=current_area_num)

        # Detect side and capture description
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

        # Save images with structured names
        for img_info in page.get_images(full=True):
            if current_area_num is None or current_side is None:
                continue
            xref    = img_info[0]
            key     = f"area_{current_area_num}_{current_side}"
            img_counter[key] = img_counter.get(key, 0) + 1
            count   = img_counter[key]

            # e.g. extracted_images/sample_doc/impacted_area_1/negative/img_1.png
            save_path = os.path.join(
                images_dir, "sample_doc",
                f"impacted_area_{current_area_num}",
                current_side,
                f"img_{count}.png"
            )
            _save_image(doc, xref, save_path)

            if current_side == "negative":
                area_map[current_area_num].negative_side_photos.append(save_path)
            else:
                area_map[current_area_num].positive_side_photos.append(save_path)

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

        imgs = page.get_images(full=True)
        # e.g. extracted_images/thermal_doc/page_1/thermal_scan.png
        page_folder = os.path.join(images_dir, "thermal_doc", f"page_{page_num + 1}")

        if len(imgs) >= 1:
            path = os.path.join(page_folder, "thermal_scan.png")
            _save_image(doc, imgs[0][0], path)
            reading.thermal_scan_path = path

        if len(imgs) >= 2:
            path = os.path.join(page_folder, "photo.png")
            _save_image(doc, imgs[1][0], path)
            reading.photo_path = path

        readings.append(reading)

    doc.close()
    return readings


# ── Combined JSON Export ───────────────────────────────────────────────────────
def extract_all_to_json(save: bool = True) -> dict:
    """
    Run both extractions, save all images, and return a unified JSON dict.
    Optionally writes the JSON to output/extracted_data.json.
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
        print(f"✓ JSON saved  → {json_path}")
        print(f"✓ Images saved → {images_dir}")

    return result


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    data = extract_all_to_json(save=True)
    print("\n=== SAMPLE REPORT ===")
    print(json.dumps(data["sample_report"], indent=2))
    print("\n=== THERMAL REPORT ===")
    print(json.dumps(data["thermal_report"], indent=2))
