"""
extract.py - PDF -> clean-ish text (the messy lane).

3GPP/ETSI PDFs carry running headers, footers, bare page numbers, and a long
table of contents. We drop the obvious junk so the chunker sees real clause
text. This is a FIRST pass: run the inspect mode and eyeball the output before
trusting it - extraction is where the mess lives.

Inspect:  python extract.py data/ts_138331v180400p.pdf
"""
import re
import sys
import fitz  # PyMuPDF

# Lines to drop.
_JUNK = [
    re.compile(r'^\s*ETSI\s*$'),
    re.compile(r'ETSI TS 1\d{2} \d{3}'),      # running header, e.g. "ETSI TS 138 331 ..."
    re.compile(r'3GPP TS \d{2}\.\d{3}'),      # version header line
    re.compile(r'^\s*\d{1,4}\s*$'),           # a bare page number on its own line
    re.compile(r'\.{3,}\s*\d+\s*$'),          # TOC entry: "... 5.3.7 title .......... 82"
    re.compile(r'^\s*Release \d+\s*$'),
]


def _keep(line: str) -> bool:
    return not any(p.search(line) for p in _JUNK)


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    kept = []
    for page in doc:
        for line in page.get_text().splitlines():
            if line.strip() and _keep(line):
                kept.append(line)
    doc.close()
    return "\n".join(kept)


if __name__ == "__main__":
    path = sys.argv[1]
    text = extract_text(path)
    print(f"extracted {len(text):,} chars from {path}\n")
    from chunker import _CLAUSE_RE
    heads = [l for l in text.splitlines() if _CLAUSE_RE.match(l.strip())]
    print(f"{len(heads)} clause headings detected; first 12:")
    for h in heads[:12]:
        print("   ", h.strip()[:72])
