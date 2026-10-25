"""report.md -> report.html (python-markdown, report_style.css) -> PDF via headless Chrome.

Usage: uv run python report/build_pdf.py report/report.md report/Shinshinova_Olga_exam_report.pdf
"""

import logging
import subprocess
import sys
from pathlib import Path

import markdown

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
logger = logging.getLogger("build_pdf")


def build(source: Path, target: Path) -> None:
    css = (source.parent / "report_style.css").read_text()
    body = markdown.markdown(source.read_text(), extensions=["tables", "sane_lists"])
    page = "@page{size:A4;margin:16mm 15mm}"
    head = f'<meta charset="utf-8"><style>{page}{css}</style>'
    html = f"<!doctype html><html><head>{head}</head><body>{body}</body></html>"
    html_path = source.with_suffix(".html")
    html_path.write_text(html)
    subprocess.run(
        [
            CHROME,
            "--headless",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={target.resolve()}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        capture_output=True,
    )
    logger.info("wrote %s", target)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build(Path(sys.argv[1]), Path(sys.argv[2]))
