"""
format_docx.py — classify paragraphs with RuBertPredictor, then apply YAML-driven formatting.

Usage (from project root):
    python format_docx.py input.docx output.docx --config format_config.yaml --batch-size 16

Element types returned by the model:  paragraph | heading | caption
Heading level is inferred from the existing Word style name (Heading 1..9 / Заголовок 1..9),
or from the leading numbering pattern when the style carries no level information.
"""

import argparse
import re
import sys
from pathlib import Path

import yaml
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.shared import Cm, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predictor import RuBertPredictor  # noqa: E402

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_ALIGNMENT = {
    "left": WD_ALIGN_PARAGRAPH.LEFT,
    "center": WD_ALIGN_PARAGRAPH.CENTER,
    "right": WD_ALIGN_PARAGRAPH.RIGHT,
    "justify": WD_ALIGN_PARAGRAPH.JUSTIFY,
}

_HEADING_STYLE_RE = re.compile(r"(heading|заголовок)\s*(\d+)", re.IGNORECASE)
_NUMBERING_RE = re.compile(r"^(\d+(?:\.\d+)*)[\.\s]")


def _heading_level_from_style(para) -> int | None:
    m = _HEADING_STYLE_RE.search(para.style.name)
    return int(m.group(2)) if m else None


def _heading_level_from_numbering(text: str) -> int:
    m = _NUMBERING_RE.match(text.strip())
    if not m:
        return 1
    return min(len(m.group(1).split(".")), 3)


def _heading_level(para) -> int:
    level = _heading_level_from_style(para)
    return level if level is not None else _heading_level_from_numbering(para.text)


def _rgb_from_hex(hex_str: str) -> RGBColor:
    h = hex_str.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _apply_font(run, cfg: dict) -> None:
    f = run.font
    if "name" in cfg:
        f.name = cfg["name"]
    if "size" in cfg:
        f.size = Pt(cfg["size"])
    if "bold" in cfg:
        f.bold = cfg["bold"]
    if "italic" in cfg:
        f.italic = cfg["italic"]
    if "color" in cfg:
        f.color.rgb = _rgb_from_hex(str(cfg["color"]))


def _apply_paragraph(para, para_cfg: dict, font_cfg: dict) -> None:
    pf = para.paragraph_format
    if "alignment" in para_cfg:
        pf.alignment = _ALIGNMENT.get(para_cfg["alignment"].lower())
    if "space_before" in para_cfg:
        pf.space_before = Pt(para_cfg["space_before"])
    if "space_after" in para_cfg:
        pf.space_after = Pt(para_cfg["space_after"])
    if "line_spacing" in para_cfg:
        pf.line_spacing = para_cfg["line_spacing"]
        pf.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
    if "first_line_indent" in para_cfg:
        pf.first_line_indent = Cm(para_cfg["first_line_indent"] or 0)
    for run in para.runs:
        _apply_font(run, font_cfg)


def _apply_config(para, element_cfg: dict) -> None:
    _apply_paragraph(para, element_cfg.get("paragraph", {}), element_cfg.get("font", {}))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def format_document(
    input_path: str,
    output_path: str,
    config_path: str,
    batch_size: int = 16,
    model_dir: str | None = None,
) -> None:
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if model_dir:
        RuBertPredictor.model_name = model_dir

    print("Loading model…")
    predictor = RuBertPredictor()

    doc = Document(input_path)
    paragraphs = [p for p in doc.paragraphs if p.text.strip()]
    print(f"Classifying {len(paragraphs)} paragraphs (batch_size={batch_size})…")

    texts = [p.text for p in paragraphs]
    labels = predictor.predict_batch(texts, batch_size=batch_size)

    counters = {"heading": 0, "caption": 0, "paragraph": 0, "skipped": 0}
    for para, label in zip(paragraphs, labels):
        if label == "heading":
            key = f"heading_{_heading_level(para)}"
            cfg = config.get(key) or config.get("heading_1")
        else:
            cfg = config.get(label)  # "caption" or "paragraph"

        if cfg:
            _apply_config(para, cfg)
            counters[label] += 1
        else:
            counters["skipped"] += 1

    doc.save(output_path)
    print(
        f"Saved → {output_path}\n"
        f"  headings: {counters['heading']}, captions: {counters['caption']}, "
        f"paragraphs: {counters['paragraph']}, skipped: {counters['skipped']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify and format a .docx using RuBertPredictor + YAML config."
    )
    parser.add_argument("input", help="Source .docx file")
    parser.add_argument("output", help="Destination .docx file")
    parser.add_argument("--config", "-c", default="format_config.yaml")
    parser.add_argument("--batch-size", "-b", type=int, default=16)
    parser.add_argument("--model", default=None, help="Override model directory")
    args = parser.parse_args()
    format_document(args.input, args.output, args.config, args.batch_size, args.model)


if __name__ == "__main__":
    main()
