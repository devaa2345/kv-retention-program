# ledger.py transfer to Machine A (A7)

Two copies of `harness/tasks/ledger.py` as it stands on Machine N:

| file | bytes | sha256 |
|---|---|---|
| `ledger.py` (as checked out on N, CRLF) | 13391 | `7694b10628f7259b878e02c66afec5d4ef69a11aa2639dfa73e5f994ed5554e6` |
| `ledger_LF.py` (LF-normalised) | see below | `ecde0d383eca2acc047b1fb42b315e4795e81cf40b3669c1ce1262240af4eddb` |

**Use `ledger_LF.py`.** The repo is checked out on Windows with git rewriting LF to CRLF, so the
raw digest of the working file is a property of the checkout, not the content.

## Line endings do NOT affect instance digests — verified, not assumed

`ledger_diff.py` refuses unless instance digests match byte for byte, so this was checked rather
than hoped:

* `EXEMPLARS` and `PREAMBLE` are built by **implicit concatenation of single-line string
  literals with explicit `\n`** (`ledger.py:60-73`), not by triple-quoted multi-line literals.
* The only triple-quoted strings in the file are **docstrings**, which never reach the generated
  context.
* Record lines come from `_fmt_record` (`ledger.py:149-150`), an f-string with no newline.
* Body assembly is `"\n".join(body_lines)` (`ledger.py:216`) — an explicit `\n` regardless of
  source encoding.

So the generated `context` string is byte-identical whether the source file is CRLF or LF, and
instance digests are invariant. Either copy will reproduce the same instances; `ledger_LF.py` is
recommended only so the *file* digest is stable across platforms.

## Generation parameters that must match

Instances in every result on Machine N were built as:

    ledger.build(seed, instance_id, target_tokens=2048, tokenizer=tok)

with `instance_id = "grid_%05d" % i`, `i` in `[0, 200)`, and `seed` from
`harness.keys.seed_key_without_seed(...)` — CRC32 of the record key, never Python `hash()`.
`N_RECORDS` and `n_bindings` (H=4) are module constants; do not override them.

The seed depends on `model` and `model_revision`, so **instances differ per model**. M2 is
`Qwen/Qwen2.5-3B-Instruct` at revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.

## One thing A should know

Payable candidate cost differs by tokenizer: **19 tokens on M2 (Qwen), 13-14 on M3 (Llama)**.
That is why M2 has no C=16 cell (VOID: k_gold = 19 > 16) and why the binding budgets differ
between models (M2 binds at C=32 and C=64; M3 at C=16 and C=32). If A's digests match but its
budget ladder disagrees, check this before suspecting the task.
