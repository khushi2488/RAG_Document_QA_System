
from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from parsers import get_all_parsers, BaseParser, ParseResult

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

DEFAULT_DATASET_DIR = "Dataset"
DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_SUMMARY_CSV = "summary.csv"

CSV_FIELDNAMES = [
    "pdf_name",
    "parser",
    "status",
    "processing_time_sec",
    "page_count",
    "char_count",
    "word_count",
    "output_file_size_bytes",
    "error_message",
]

logger = logging.getLogger("compare_parsers")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def discover_pdfs(dataset_dir: Path) -> list[Path]:
    """Return every *.pdf file directly inside dataset_dir, sorted by name."""
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_dir.resolve()}\n"
            f"Create it and add PDF files, e.g.: mkdir {dataset_dir}"
        )
    return sorted(p for p in dataset_dir.glob("*.pdf") if p.is_file())


def count_words(text: str) -> int:
    """Whitespace-based word count -- simple, fast, and format-agnostic."""
    return len(text.split())


def select_parsers(all_parsers: list[BaseParser], only: Optional[str]) -> list[BaseParser]:
    """Filter the full parser list down to a comma-separated subset of names."""
    if not only:
        return all_parsers
    wanted = {name.strip().lower() for name in only.split(",") if name.strip()}
    known_names = {p.name for p in all_parsers}
    unknown = wanted - known_names
    if unknown:
        raise ValueError(
            f"Unknown parser name(s): {sorted(unknown)}. "
            f"Known parsers: {', '.join(sorted(known_names))}"
        )
    return [p for p in all_parsers if p.name in wanted]


def format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}s"


# --------------------------------------------------------------------------- #
# Core benchmarking logic
# --------------------------------------------------------------------------- #

def benchmark_single(pdf_path: Path, parser: BaseParser, output_dir: Path) -> dict:
    """
    Run a single parser against a single PDF, save its output to disk, and
    return one row (dict) of metrics ready to write to the summary CSV.
    """
    row = {
        "pdf_name": pdf_path.name,
        "parser": parser.name,
        "status": "failed",
        "processing_time_sec": 0.0,
        "page_count": 0,
        "char_count": 0,
        "word_count": 0,
        "output_file_size_bytes": 0,
        "error_message": "",
    }

    if not parser.is_available():
        row["status"] = "skipped"
        row["error_message"] = "Dependency not installed"
        logger.warning("  [%s] skipped: dependency not installed", parser.name)
        return row

    out_path = output_dir / parser.name / f"{pdf_path.stem}{parser.output_ext}"

    start = time.perf_counter()
    try:
        result: ParseResult = parser.parse(pdf_path)
    except Exception as exc:
        # Defensive fallback: parsers are contractually required to catch
        # their own exceptions, but a bug there shouldn't take down the run.
        elapsed = time.perf_counter() - start
        row["processing_time_sec"] = round(elapsed, 4)
        row["error_message"] = f"Unhandled {type(exc).__name__}: {exc}"
        logger.error("  [%s] crashed on %s: %s", parser.name, pdf_path.name, exc)
        return row
    elapsed = time.perf_counter() - start
    row["processing_time_sec"] = round(elapsed, 4)

    if not result.success:
        row["error_message"] = result.error
        logger.error("  [%s] failed on %s: %s", parser.name, pdf_path.name, result.error)
        return row

    # Persist the extracted text so extraction *quality* -- not just the
    # timing numbers -- can be inspected manually afterwards.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.text, encoding="utf-8")

    row.update(
        status="success",
        page_count=result.page_count,
        char_count=len(result.text),
        word_count=count_words(result.text),
        output_file_size_bytes=out_path.stat().st_size,
    )
    logger.info(
        "  [%s] %s -- %s, %d pages, %d chars",
        parser.name, pdf_path.name, format_seconds(elapsed), result.page_count, len(result.text),
    )
    return row


def run_benchmark(
    dataset_dir: Path,
    output_dir: Path,
    summary_csv: Path,
    parser_filter: Optional[str] = None,
) -> list[dict]:
    """Run every selected parser against every PDF and write summary.csv."""
    pdfs = discover_pdfs(dataset_dir)
    if not pdfs:
        logger.warning("No PDF files found in %s -- nothing to do.", dataset_dir.resolve())
        return []

    parsers = select_parsers(get_all_parsers(), parser_filter)
    if not parsers:
        logger.warning("No parsers selected -- nothing to do.")
        return []

    logger.info("Found %d PDF(s) in %s", len(pdfs), dataset_dir)
    logger.info("Running parsers: %s", ", ".join(p.name for p in parsers))

    for parser in parsers:
        if not parser.is_available():
            logger.warning(
                "Parser '%s' is not available (dependency not installed) -- "
                "it will be recorded as 'skipped' for every PDF.", parser.name,
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for pdf_path in pdfs:
        logger.info("Processing %s", pdf_path.name)
        for parser in parsers:
            rows.append(benchmark_single(pdf_path, parser, output_dir))

    write_summary_csv(rows, summary_csv)
    print_summary_table(rows)
    return rows


def write_summary_csv(rows: list[dict], summary_csv: Path) -> None:
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), summary_csv.resolve())


# --------------------------------------------------------------------------- #
# Console report
# --------------------------------------------------------------------------- #

def print_summary_table(rows: list[dict]) -> None:
    """Print a compact, human-readable aggregate (per parser) to stdout."""
    if not rows:
        return

    by_parser: dict[str, list[dict]] = {}
    for row in rows:
        by_parser.setdefault(row["parser"], []).append(row)

    header = (
        f"{'Parser':<14}{'Success':<10}{'Failed':<9}{'Skipped':<9}"
        f"{'Avg Time':<12}{'Avg Pages':<11}{'Avg Chars':<12}{'Avg Words':<10}"
    )
    print("\n" + "=" * len(header))
    print("BENCHMARK SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for parser_name, parser_rows in by_parser.items():
        succeeded = [r for r in parser_rows if r["status"] == "success"]
        failed = [r for r in parser_rows if r["status"] == "failed"]
        skipped = [r for r in parser_rows if r["status"] == "skipped"]

        n = len(succeeded) or 1  # avoid div-by-zero; averages are 0 anyway if n==0
        avg_time = sum(r["processing_time_sec"] for r in succeeded) / n if succeeded else 0.0
        avg_pages = sum(r["page_count"] for r in succeeded) / n if succeeded else 0.0
        avg_chars = sum(r["char_count"] for r in succeeded) / n if succeeded else 0.0
        avg_words = sum(r["word_count"] for r in succeeded) / n if succeeded else 0.0

        print(
            f"{parser_name:<14}{len(succeeded):<10}{len(failed):<9}{len(skipped):<9}"
            f"{avg_time:<12.3f}{avg_pages:<11.1f}{avg_chars:<12.1f}{avg_words:<10.1f}"
        )

    print("=" * len(header) + "\n")
    print(f"Full per-file results written to summary.csv\n")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark PyMuPDF, PDFPlumber, Unstructured and Docling on a folder of PDFs.",
    )
    parser.add_argument(
        "--dataset-dir", default=DEFAULT_DATASET_DIR,
        help=f"Folder containing input PDFs (default: {DEFAULT_DATASET_DIR})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Folder to write per-parser extracted text into (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--summary-csv", default=DEFAULT_SUMMARY_CSV,
        help=f"Path to write the metrics CSV to (default: {DEFAULT_SUMMARY_CSV})",
    )
    parser.add_argument(
        "--parsers", default=None,
        help="Comma-separated subset of parsers to run, e.g. 'pymupdf,pdfplumber'. "
             "Default: run all four.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Configure our own logger explicitly rather than calling
    # logging.basicConfig() at DEBUG level, which would also crank up
    # verbose internal loggers from pdfminer/pdfplumber/etc. to DEBUG and
    # flood the console with unrelated parsing internals.
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.propagate = False

    try:
        run_benchmark(
            dataset_dir=Path(args.dataset_dir),
            output_dir=Path(args.output_dir),
            summary_csv=Path(args.summary_csv),
            parser_filter=args.parsers,
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
