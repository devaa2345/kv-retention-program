"""Assemble PREREG_P3.md from its template plus the generated prediction tables.

One file, one hash. The tables are spliced in rather than referenced so that the frozen hash
covers the committed numbers themselves, not a pointer to them.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
tpl = (HERE / "PREREG_P3.template.md").read_text(encoding="utf-8")
tab = (HERE / "out" / "stage4_predictions.md").read_text(encoding="utf-8")
marker = "<!--STAGE4_PREDICTIONS-->"
assert marker in tpl, "template marker missing"
out = tpl.replace(marker, tab.rstrip(chr(10)))
(HERE / "PREREG_P3.md").write_text(out, encoding="utf-8", newline="\n")
print("wrote PREREG_P3.md  (%d lines)" % len(out.split(chr(10))))
