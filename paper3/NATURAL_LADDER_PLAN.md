# Natural-text evidence ladder (Experiment A) - gate plan, frozen before any GPU run

Dataset: data/natural/nat_v1.jsonl (sha256 in nat_v1.jsonl.sha256), D=32, H=4, N=36, 200 instances,
L = 4096 +/- 32 in EACH tokenizer (separate rendering per tokenizer: Qwen counts ~190 more tokens
than Llama on the same text because of digit tokenisation, so one text cannot satisfy both).
Prose: 8 US-public-domain Project Gutenberg books. Facts: one generated template family.

Gate, n = 100 (first 100 instances), M2 and M3:
 1. full_cache in [0.55, 0.97] at each cost level (1, 3, 5 elements).
 2. floor_pos > 0.05 at EVERY (level, budget) cell, budgets C in {256, 512, 1024}
    (6/12.5/25% of context = Paper 3's 128/256/512 at L=2048). C=128 is excluded in advance:
    its analytic ceiling for floor_pos is 0.051-0.054 with a perfect reader, so >0.05 is
    unreachable by construction. Ceilings at 256/512/1024 are 0.089/0.152/0.285.
 3. Probe 4: full_cache with the queried fact deleted <= 1/(H+D)+0.02 = 0.0478 at each level.
Single permitted adjustment: distractor count D only (competence < 0.55 -> D=20; > 0.97 -> D=40).
A floor failure that D cannot move (structural tail share) is reported, not adjusted.
Second failure -> stop and report "not constructible at these scales". No third build.
