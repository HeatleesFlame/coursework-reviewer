"""
bench.py — measure inference speed on a .docx file.

Usage (from project root):
    python bench.py coursework_template_standard.docx
    python bench.py coursework_template_standard.docx --batch-sizes 1,8,16,32
"""

import argparse
import sys
import time
from pathlib import Path

from docx import Document

sys.path.insert(0, str(Path(__file__).parent / "src"))
from predictor import RuBertPredictor


def bench(docx_path: str, batch_sizes: list[int]) -> None:
    texts = [p.text for p in Document(docx_path).paragraphs if p.text.strip()]
    print(f"Document: {docx_path}")
    print(f"Paragraphs: {len(texts)}\n")

    print("Loading model…")
    predictor = RuBertPredictor()

    # sequential (batch_size=1)
    if 1 in batch_sizes:
        t0 = time.perf_counter()
        for text in texts:
            predictor.predict(text)
        elapsed = time.perf_counter() - t0
        print(f"  sequential (predict one-by-one): {elapsed:.2f}s  ({elapsed/len(texts)*1000:.1f} ms/para)")
        batch_sizes = [b for b in batch_sizes if b != 1]

    for bs in batch_sizes:
        t0 = time.perf_counter()
        predictor.predict_batch(texts, batch_size=bs)
        elapsed = time.perf_counter() - t0
        print(f"  batch_size={bs:<4}: {elapsed:.2f}s  ({elapsed/len(texts)*1000:.1f} ms/para)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark RuBertPredictor on a .docx file.")
    parser.add_argument("docx", help=".docx file to benchmark on")
    parser.add_argument(
        "--batch-sizes",
        default="1,8,16,32,64",
        help="Comma-separated batch sizes to test (default: 1,8,16,32,64)",
    )
    args = parser.parse_args()
    sizes = [int(x) for x in args.batch_sizes.split(",")]
    bench(args.docx, sizes)


if __name__ == "__main__":
    main()
