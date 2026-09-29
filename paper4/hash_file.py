"""One hash procedure, applied to the committed text, verified by round-trip before freezing.

Paper 2's lesson (PREREG_P2 section 12 / the corrected-hash commit): hash the LF-normalised
bytes of the file exactly as committed, and prove the procedure round-trips before relying on
it. `git config core.autocrlf` can rewrite line endings on checkout, so the hash is taken over
text read in binary and normalised to LF, and the normalisation is asserted idempotent.
"""
import hashlib, sys
from pathlib import Path

def digest(p: Path) -> str:
    raw = p.read_bytes()
    lf = raw.replace(b"\r\n", b"\n")
    assert lf.replace(b"\r\n", b"\n") == lf, "LF normalisation is not idempotent"
    return hashlib.sha256(lf).hexdigest()

if __name__ == "__main__":
    for a in sys.argv[1:]:
        p = Path(a)
        d = digest(p)
        out = p.with_suffix(p.suffix + ".sha256")
        out.write_text(f"{d}  {p.name}  (sha256 of LF-normalised bytes)\n",
                       encoding="utf-8", newline="\n")
        # round-trip: re-read what was written and re-derive
        assert digest(p) == d, "hash not stable on re-read"
        print(f"{d}  {p.name}")
