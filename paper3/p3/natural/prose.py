"""Real surrounding prose: cleaned Project Gutenberg paragraphs (US public domain).

Licence record: every book below is an English-language work first published before 1929 and
distributed by Project Gutenberg as public domain in the USA. The Gutenberg licence header and
trademark boilerplate are STRIPPED, so no Gutenberg trademark text is redistributed. Only prose
paragraphs are used; headings, tables, verse, footnotes and lines with heavy digits are dropped.
"""
from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "data" / "natural" / "sources"
BOOKS = {  # id -> (title, author, first published)
    2009: ("The Origin of Species", "Charles Darwin", 1859),
    3300: ("The Wealth of Nations", "Adam Smith", 1776),
    34901: ("On Liberty", "John Stuart Mill", 1859),
    5827: ("The Problems of Philosophy", "Bertrand Russell", 1912),
    148: ("The Autobiography of Benjamin Franklin", "Benjamin Franklin", 1791),
    205: ("Walden", "Henry David Thoreau", 1854),
    2944: ("Essays: First Series", "Ralph Waldo Emerson", 1841),
    7370: ("Second Treatise of Government", "John Locke", 1689),
}
_SENT = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(]?[A-Z])")
_ABBR = re.compile(r"\b(?:Mr|Mrs|Dr|St|Mt|vs|Messrs|Col|Capt|Gen|Prof|Hon|Rev|No|viz|cf|i\.e|e\.g)\.$")


def _body(text: str) -> str:
    a = text.index("*** START OF")
    a = text.index("\n", a) + 1
    b = text.index("*** END OF") if "*** END OF" in text else len(text)
    return text[a:b].replace("\r", "")


def sentences(par: str) -> list[str]:
    out, cur = [], ""
    for piece in _SENT.split(par):
        cur = (cur + " " + piece).strip() if cur else piece
        if _ABBR.search(cur):
            continue
        out.append(cur)
        cur = ""
    if cur:
        out.append(cur)
    return out


def paragraphs(book_id: int) -> list[str]:
    raw = _body((SRC / f"pg{book_id}.txt").read_text(encoding="utf-8"))
    pars = []
    for blk in re.split(r"\n\s*\n", raw):
        lines = [l.strip() for l in blk.split("\n") if l.strip()]
        if not lines:
            continue
        p = " ".join(lines)
        words = p.split()
        alpha = sum(c.isalpha() for c in p) / max(1, len(p))
        digits = sum(c.isdigit() for c in p) / max(1, len(p))
        if not (50 <= len(words) <= 230) or alpha < 0.78 or digits > 0.02:
            continue
        if p.isupper() or "[" in p or "]" in p or "_" in p or "Gutenberg" in p or "http" in p:
            continue
        if not re.match(r"^[\"'“‘(]?[A-Z]", p) or not re.search(r"[.!?][\"'”’)]?$", p):
            continue
        if any(ord(c) > 0x2019 or c in " —" and False for c in p):
            pass
        p = re.sub(r"\s+", " ", p)
        # normalise to plain ASCII punctuation so char offsets are tokenizer-independent text
        p = (p.replace("’", "'").replace("‘", "'").replace("“", '"')
              .replace("”", '"').replace("—", " - ").replace("–", "-"))
        if any(ord(c) > 127 for c in p):
            continue
        if len(sentences(p)) < 2:
            continue
        pars.append(p)
    return pars


def all_paragraphs() -> dict[int, list[str]]:
    return {b: paragraphs(b) for b in BOOKS}
