"""Per-instance fact geometry for Paper 2's LEDGER instances, rebuilt on CPU.

Paper 2's dumps record aggregate keep statistics per (instance, C, arm) but not the token
counts the statistics are fractions *of*. Those counts are deterministic functions of the
instance seed, so they are rebuilt here rather than re-measured on GPU.

Validation: the rebuilt templated context length is compared token-for-token against the
`n_ctx` recorded in the Paper 2 grid rows. If that does not match exactly the cache refuses
to write, because a mismatch means the instance is not the one that was run.

Emits, per model, per instance:
    L              templated context length in tokens (== recorded n_ctx)
    q_line         list of H=4 queried-record LINE token counts
    q_idval        list of H=4 queried-record (id, value) token counts
    all_line       all 40 record LINE token counts
    all_idval      all 40 record (id, value) token counts
    q_line_floor   per queried record, tokens of it already inside the mandatory floors
                   (first n_sink, last n_window) -- these are free to every arm
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

P2 = Path(__file__).resolve().parents[2] / "paper2"
sys.path.insert(0, str(P2))

MODELS = {
    "M2": "Qwen/Qwen2.5-3B-Instruct",
    "M3": "meta-llama/Llama-3.2-3B-Instruct",
}
GRID = {
    "M2": P2 / "runs/nvidia/grid_M2_ledger_agnostic.jsonl",
    "M3": P2 / "runs/nvidia/grid_M3_ledger_agnostic.jsonl",
}
N_SINK, N_WINDOW = 8, 64
REC = re.compile(r"^R(\d{3}) \| ")
CACHE = Path(__file__).resolve().parents[1] / "out" / "facts_cache.json"


def templated_parts(tok, context: str, query: str):
    """Verbatim from paper2/stage5_ladder_validation.py (imported there via kvpress)."""
    marker = "␟QUERY␟"
    full = tok.apply_chat_template(
        [{"role": "user", "content": context + "\n\n" + marker}],
        tokenize=False, add_generation_prompt=True)
    pre, post = full.split(marker)
    return pre, query + post


def seeds_and_nctx(path: Path):
    seeds, nctx = {}, {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            k = r["key"]
            seeds.setdefault(k["instance_id"], k["seed"])
            nctx.setdefault(k["instance_id"], r.get("n_ctx"))
    return seeds, nctx


def build(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text(encoding="utf-8"))

    from transformers import AutoTokenizer
    from harness.tasks import ledger

    out = {}
    for tag, M in MODELS.items():
        tok = AutoTokenizer.from_pretrained(M)
        seeds, nctx = seeds_and_nctx(GRID[tag])
        per = {}
        for iid in sorted(seeds):
            inst = ledger.build(seeds[iid], iid, target_tokens=2048, tokenizer=tok)
            pre, _ = templated_parts(tok, inst.context, "")
            enc = tok(pre, add_special_tokens=False, return_offsets_mapping=True)
            off = enc["offset_mapping"]
            L = len(off)
            if nctx[iid] is not None and L != nctx[iid]:
                raise SystemExit(
                    f"{tag} {iid}: rebuilt n_ctx {L} != recorded {nctx[iid]}; refusing to cache")
            base = pre.index(inst.context)
            floor = set(range(min(N_SINK, L))) | set(range(max(0, L - N_WINDOW), L))
            lines, idvals = {}, {}
            for line in inst.context.split("\n"):
                m = REC.match(line)
                if not m or not (1 <= int(m.group(1)) <= ledger.N_RECORDS):
                    continue
                a = inst.context.index(line) + base
                b = a + len(line)
                full = {ti for ti, (x, y) in enumerate(off) if y > x and x < b and y > a}
                ia, ib = a, a + 4
                va, vb = b - 6, b
                idv = {ti for ti, (x, y) in enumerate(off)
                       if y > x and ((x < ib and y > ia) or (x < vb and y > va))}
                lines[m.group(0)] = full
                idvals[m.group(0)] = idv
            qids = [v.rec_id + " | " for v in inst.variants]
            per[iid] = dict(
                L=L,
                q_line=[len(lines[q]) for q in qids],
                q_idval=[len(idvals[q]) for q in qids],
                q_line_floor=[len(lines[q] & floor) for q in qids],
                q_idval_floor=[len(idvals[q] & floor) for q in qids],
                all_line=[len(s) for s in lines.values()],
                all_idval=[len(s) for s in idvals.values()],
                n_records=len(lines),
            )
        out[tag] = dict(model=M, n_sink=N_SINK, n_window=N_WINDOW, instances=per)
        print(f"{tag}: {len(per)} instances rebuilt, n_ctx matched exactly", file=sys.stderr)

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(out), encoding="utf-8")
    return out


if __name__ == "__main__":
    d = build(force="--force" in sys.argv)
    import statistics as st
    for tag, m in d.items():
        ins = m["instances"]
        ql = [x for v in ins.values() for x in v["q_line"]]
        qi = [x for v in ins.values() for x in v["q_idval"]]
        al = [x for v in ins.values() for x in v["all_line"]]
        fl = [x for v in ins.values() for x in v["q_line_floor"]]
        print(f"{tag} {m['model']}")
        print(f"   n={len(ins)}  L mean {st.fmean(v['L'] for v in ins.values()):.1f}"
              f"  [{min(v['L'] for v in ins.values())}, {max(v['L'] for v in ins.values())}]")
        print(f"   c_line  queried: mean {st.fmean(ql):.4f}  all-40: mean {st.fmean(al):.4f}")
        print(f"   c_idval queried: mean {st.fmean(qi):.4f}")
        print(f"   queried-record tokens already inside the floors: mean {st.fmean(fl):.4f}")
