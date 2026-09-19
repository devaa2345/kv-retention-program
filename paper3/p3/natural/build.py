"""Assemble one natural-text instance and record every span at construction time.

Order of operations (each step is asserted, not assumed):
  1. draw real paragraphs (public-domain prose) and split into sentences;
  2. generate N = H + D fact sentences (one template family, lexically matched);
  3. pick N insertion slots by STRATIFIED jitter on the character axis (one slot per stride,
     jittered inside the middle half of the stride), snapped to prose sentence boundaries;
  4. put the H queried facts one per equal group of slots, so gold is spread over the whole
     context and cannot cluster (Paper 2 v1 failed exactly this way: contiguous block ->
     floor_pos absorbing at 0.0000);
  5. render, recording the character span of every inserted sentence and every element;
  6. search the prose length until BOTH tokenizers give 4096 +/- 32 for the SAME text;
  7. map every char span to token indices per tokenizer, from offset mappings.
"""
from __future__ import annotations

import random
import re
import zlib

from p3.natural import facts as F
from p3.natural import prose as P

TARGET, TOL = 4096, 32
H = 4
_WORD = re.compile(r"[A-Za-z]+")


def seed_for(version: str, iid: str) -> int:
    return zlib.crc32(f"{version}|{iid}".encode()) & 0x7FFFFFFF


def _render(items):
    """items: list of dict(kind, text, par_start). Returns text and per-item (start, end)."""
    out, spans, at = [], [], 0
    for i, it in enumerate(items):
        sep = "" if i == 0 else ("\n\n" if (it["kind"] == "prose" and it["par_start"]) else " ")
        out.append(sep)
        at += len(sep)
        spans.append((at, at + len(it["text"])))
        out.append(it["text"])
        at += len(it["text"])
    return "".join(out), spans


def _slots(rng, sent_cum, n_sent, N):
    """N distinct insertion indices (insert BEFORE prose sentence j, 1<=j<n_sent), one per
    stride of the cumulative character axis, jittered inside the middle half of the stride."""
    total = sent_cum[-1]
    idx = []
    for s in range(N):
        target = (s + rng.uniform(0.25, 0.75)) / N * total
        j = min(range(1, n_sent), key=lambda k: abs(sent_cum[k] - target))
        idx.append(j)
    for s in range(1, N):                       # keep strictly increasing
        if idx[s] <= idx[s - 1]:
            idx[s] = idx[s - 1] + 1
    if idx[-1] >= n_sent:
        return None
    return idx


def build(version: str, iid: str, D: int, tok_by_tag: dict, paras: dict, max_attempts: int = 400):
    seed = seed_for(version, iid)
    N = H + D
    pre = F.preamble()
    pool = [(b, i, p) for b, ps in paras.items() for i, p in enumerate(ps)]
    for attempt in range(max_attempts):
        rng = random.Random(seed + 7919 * attempt)
        chosen = rng.sample(pool, 90)
        sents = []                                   # (text, par_start, book, para_idx)
        for b, i, p in chosen:
            for k, s in enumerate(P.sentences(p)):
                sents.append((s, k == 0, b, i))
        words = {w.lower() for s in sents for w in _WORD.findall(s[0])}
        facts = F.make_facts(rng, N, words)
        order = list(range(N))                        # slot -> fact index
        rng.shuffle(order)
        grp = N // H
        qslots = sorted(rng.randrange(g * grp, (g + 1) * grp if g < H - 1 else N)
                        for g in range(H))
        def trial(m):
            keep = sents[:m]
            cum = [0]
            for s_ in keep:
                cum.append(cum[-1] + len(s_[0]) + 1)
            slots = _slots(random.Random(seed + attempt * 31 + m), cum, len(keep), N)
            if slots is None:
                return None
            items, at_slot, fact_item = [], dict(zip(slots, range(N))), {}
            for j, (s_, ps, b_, pi) in enumerate(keep):
                if j in at_slot:
                    fi = order[at_slot[j]]
                    fact_item[fi] = len(items)
                    items.append(dict(kind="fact", text=facts[fi].text(), par_start=False))
                items.append(dict(kind="prose", text=s_, par_start=ps and j > 0))
            text, spans = _render(items)
            ctx = pre + text + "\n"
            return keep, slots, items, spans, fact_item, ctx

        rends, ok = {}, True
        for tag, tk in tok_by_tag.items():
            def count(m, tk=tk):
                r = trial(m)
                return None if r is None else (r, len(tk(r[5], add_special_tokens=False)["input_ids"]))
            lo, hi = N + 5, len(sents)
            while lo < hi:                            # smallest m with count >= TARGET
                mid = (lo + hi) // 2
                r = count(mid)
                if r is None or r[1] < TARGET:
                    lo = mid + 1
                else:
                    hi = mid
            found = None
            for m in range(max(N + 5, lo - 6), min(len(sents), lo + 6) + 1):
                r = count(m)
                if r is not None and abs(r[1] - TARGET) <= TOL:
                    found = (r[0], r[1])
                    break
            if found is None:
                ok = False
                break
            rends[tag] = found
        if ok:
            return _finish(version, iid, D, seed, attempt, pre, facts, order, qslots, rends,
                           tok_by_tag)
    raise RuntimeError(f"{iid}: no text within {TARGET}+/-{TOL} for all tokenizers")


def _tokmap(tk, ctx, ranges):
    enc = tk(ctx, add_special_tokens=False, return_offsets_mapping=True)
    off = enc["offset_mapping"]
    n = len(off)

    def tokens(a, b):
        ids = [i for i, (x, y) in enumerate(off) if y > x and x < b and y > a]
        return [ids[0], ids[-1] + 1] if ids else None
    return n, {k: tokens(a, b) for k, (a, b) in ranges.items()}


def _finish(version, iid, D, seed, attempt, pre, facts, order, qslots, rends, tok_by_tag):
    N = H + D
    qfacts = [order[s] for s in qslots]
    ex = set(F.EXEMPLAR.values()) | {"Wexbourne"}
    for f in facts:
        assert not ({f.prog, f.person, f.loc} & ex)
    variants = []
    for fi in qfacts:
        f = facts[fi]
        for lv in F.LEVELS:
            variants.append(dict(fact_id=f.fact_id, level=lv, query=F.query_text(lv, f.prog),
                                 elements=F.element_answers(f, lv),
                                 elements_keys=list(F.LEVELS[lv])))
    renderings = {}
    for tag, ((keep, slots, items, spans, fact_item, ctx), n_tok) in rends.items():
        tk = tok_by_tag[tag]
        base, total = len(pre), len(ctx)
        frecs = []
        for fi, f in enumerate(facts):
            a, b = spans[fact_item[fi]]
            a += base
            b += base
            el = {k: (a + s_, a + e_) for k, (s_, e_) in f.spans().items()}
            gold = {lv: (a, el[F.LEVELS[lv][-1]][1]) for lv in F.LEVELS}
            assert ctx[a:b] == f.text()
            rg = {"sentence": (a, b), "cue": el["prog"]}
            rg.update({"el_" + k: v for k, v in el.items()})
            rg.update({"gold_%d" % lv: v for lv, v in gold.items()})
            _, tokmap = _tokmap(tk, ctx, rg)
            frecs.append(dict(fact_id=f.fact_id, role="queried" if fi in qfacts else "distractor",
                              slot=order.index(fi), char=[a, b], text=f.text(),
                              elements={k: dict(text=getattr(f, k), char=list(v))
                                        for k, v in el.items()},
                              gold_char={str(lv): list(v) for lv, v in gold.items()},
                              token_ranges=tokmap))
        starts = sorted(r["char"][0] for r in frecs)
        span = total - len(pre)
        stride = span / N
        gaps = [y - x for x, y in zip(starts, starts[1:])]
        spread = dict(first_frac=(starts[0] - base) / span, last_frac=(starts[-1] - base) / span,
                      min_gap_over_stride=min(gaps) / stride,
                      max_gap_over_stride=max(gaps) / stride,
                      queried_frac=sorted(round((r["char"][0] - base) / span, 4)
                                          for r in frecs if r["role"] == "queried"))
        assert spread["first_frac"] < 2.0 / N and spread["last_frac"] > 1 - 2.5 / N, spread
        assert spread["max_gap_over_stride"] < 3.4, spread
        renderings[tag] = dict(context=ctx, n_chars=total, n_tokens=n_tok, facts=frecs,
                               spread=spread,
                               prose_sentences_kept=len(keep),
                               prose_sources=sorted({(b_, pi) for (_, _, b_, pi) in keep}))
    return dict(version=version, instance_id=iid, seed=seed, attempt=attempt, H=H, D=D, N=N,
                preamble_chars=len(pre), variants=variants, persons=[f.person for f in facts],
                renderings=renderings)
