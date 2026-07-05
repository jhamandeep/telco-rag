"""
chunker.py - spec-structure-aware chunker for the TelcoRAG ingest.

This module is what the k3s ingest Job imports. It turns extracted spec text
into a list of Chunk records {id, text, metadata}, ready for embedding + upsert
into the `telco_ran` Qdrant collection.

Two modes, selected by CHUNK_MODE (set in the ingest ConfigMap, never a .env):

  "fixed"      baseline. Blind fixed-length character windows with overlap.
               No structural awareness. Produces the MEASURED BASELINE.

  "structure"  cut on the spec's own clause hierarchy; keep each clause whole;
               sub-split only over-long clauses, WITH overlap; tag metadata
               {spec, clause, type, release} on every chunk.

PDF text extraction lives in extract.py (container-side; needs PyMuPDF).
This module is pure-Python (regex + stdlib only) so its logic can be validated
in isolation, without PDFs or the cluster, before we containerize it.
"""

import re
import hashlib
from dataclasses import dataclass, asdict

# --- clause heading detection -------------------------------------------------
# 3GPP clauses look like:  "5.3.7 RRC connection re-establishment"
#                          "5.3.7.1 General"
# Heading = dotted number at line start, then a Capitalised title, kept short.
# This is a first-pass heuristic; refine the title bound against real specs.
_CLAUSE_RE = re.compile(r'^(?P<num>\d+(?:\.\d+){0,5})\s+(?P<title>[A-Z][^\n]{0,80})$')

# Type heuristics. NR RRC timers are T3xx (T300/T301/T304/T310/T311/T319 ...).
_TIMER_RE = re.compile(r'\bT3\d{2}\b')
_MEAS_RE  = re.compile(r'\bRSR[PQ]\b|\bSINR\b|\bEvent\s+A\d\b', re.IGNORECASE)


def classify(text: str) -> str:
    """First-pass clause-type tag. Order matters: most specific wins."""
    if 'ASN1START' in text or '::=' in text:
        return 'asn1'
    if _TIMER_RE.search(text) or 'timer' in text.lower():
        return 'timer'
    if _MEAS_RE.search(text):
        return 'measurement'
    if 'procedure' in text.lower():
        return 'procedure'
    return 'text'


@dataclass
class Chunk:
    spec: str            # e.g. "38.331"  (from the source filename)
    clause: str          # e.g. "5.3.7.1" ('?' in fixed mode - no structure)
    ctype: str           # timer | measurement | procedure | asn1 | text
    text: str
    n_chars: int = 0
    cid: str = ''
    release: str = 'Rel-17'

    def finalize(self) -> 'Chunk':
        self.text = self.text.strip()
        self.n_chars = len(self.text)
        key = f"{self.spec}|{self.clause}|{self.text[:64]}"
        self.cid = hashlib.sha1(key.encode()).hexdigest()[:16]
        return self

    def payload(self) -> dict:
        """What gets stored alongside the vector in Qdrant."""
        return asdict(self)


# --- mode 1: fixed (baseline) -------------------------------------------------
def chunk_fixed(text, spec, size=900, overlap=120, release='Rel-17'):
    """Baseline: blind character windows. No clause awareness, no real metadata.
    Mirrors the current pipeline's CHUNK_SIZE=900 behaviour."""
    text = re.sub(r'[ \t]+', ' ', text).strip()
    chunks, i, step = [], 0, max(1, size - overlap)
    while i < len(text):
        window = text[i:i + size]
        chunks.append(
            Chunk(spec=spec, clause='?', ctype=classify(window),
                  text=window, release=release).finalize()
        )
        i += step
    return chunks


# --- mode 2: structure-aware --------------------------------------------------
def split_clauses(text):
    """Split into (clause_number, clause_title, body) on heading lines."""
    out, num, title, buf = [], '0', 'Preamble', []
    for line in text.splitlines():
        m = _CLAUSE_RE.match(line.strip())
        if m:
            if buf:
                out.append((num, title, '\n'.join(buf)))
            num, title, buf = m.group('num'), m.group('title'), [line]
        else:
            buf.append(line)
    if buf:
        out.append((num, title, '\n'.join(buf)))
    return out


def chunk_structure(text, spec, max_chars=1400, overlap=150, release='Rel-17'):
    """One sub-clause = one chunk. Over-long clauses sub-split WITH overlap so a
    definition spanning the cut survives. Every chunk carries clause + type."""
    chunks = []
    for num, title, body in split_clauses(text):
        body = body.strip()
        if not body:
            continue
        ctype = classify(body)
        if len(body) <= max_chars:
            chunks.append(Chunk(spec, num, ctype, body, release=release).finalize())
        else:
            step = max(1, max_chars - overlap)
            for i in range(0, len(body), step):
                piece = body[i:i + max_chars]
                chunks.append(Chunk(spec, num, ctype, piece, release=release).finalize())
    return chunks


# --- entry point used by the ingest Job --------------------------------------
def chunk_document(text, spec, mode='structure', release='Rel-17', **kw):
    if mode == 'fixed':
        return chunk_fixed(text, spec, release=release, **kw)
    if mode == 'structure':
        return chunk_structure(text, spec, release=release, **kw)
    raise ValueError(f"unknown CHUNK_MODE: {mode!r} (expected 'fixed' or 'structure')")
