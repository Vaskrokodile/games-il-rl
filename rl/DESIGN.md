# Society-Game Reasoning RL Pipeline — M1 Air (8GB) Feasible Edition

> Status: **design only**. No code yet.
>
> **Hardware target: Apple M1 Air, 8GB unified memory, Mac-only, no external
> compute.** This constraint drives every design choice below. A "scaling to
> multi-GPU" appendix (Section 15) preserves the original larger design for when
> bigger compute is available.
>
> **One-line summary:** Train a **0.5B model** with **LoRA + GRPO** via
> **Apple MLX**, using **rule-based society-game opponents only**, at
> **proof-of-concept scale**, to test whether society-game RL transfers to
> **more token-efficient math reasoning**.

---

## 1. Hypothesis & rationale

**Core claim.** Reasoning quality and reasoning *efficiency* are partly
metacognitive skills, not purely domain skills. A model that learns to take the
right decision *fast* under adversarial, uncertain, multi-agent pressure will
develop a transferable habit of **committing to correct decisions without
wasteful self-reversal** ("oh wait, but actually..."). This habit should
transfer to math/logical reasoning, yielding equal-or-better accuracy at
**fewer tokens**.

**Why society games are the right gym:** uncertainty + adversarial pressure
punish over-deliberation; dense unambiguous reward (win/loss) avoids fragile
verifiers; theory-of-mind is a generalizable primitive; games give abundance +
natural curriculum; held-out games test reasoning-vs-memorization.

**What this 0.5B run can and cannot prove:**
- **Can:** show whether the *mechanism* plausibly exists — does society-game RL
  with an efficiency/consistency reward reduce self-reversals and tokens on math
  vs a math-only-RL baseline *at 0.5B scale*?
- **Cannot:** prove it scales to 1.5B+ or that absolute math accuracy is
  competitive. A **null result here is inconclusive** (could be scale, not
  mechanism). A **positive result is a meaningful green light** to scale up.

---

## 2. Model & framework (M1 Air feasible)

**Training base:** `Qwen/Qwen2.5-0.5B-Instruct`
- Why not `DeepSeek-R1-Distill-Qwen-1.5B`: 1.5B full-FT optimizer states alone
  (~12GB) exceed 8GB; even LoRA on 1.5B on 8GB is at the OOM edge and too slow
  for RL rollout volumes. 0.5B is the largest model that leaves real headroom
  for LoRA training + sequential rollouts on 8GB unified.
- No 0.5B R1 distill exists, so we **inject R1-style reasoning format via the
  SFT warmup** (Section 7) — this is on-thesis: the warmup teaches the
  reasoning trace format; RL then shapes *how* it reasons.

**Optional mentor (inference only, used sparingly):**
- `DeepSeek-R1-Distill-Qwen-1.5B` at **Q4 (~1GB)** via MLX, used *offline, a
  handful of times* to generate a small set of expert game traces for SFT.
  Not in the RL loop (too slow). If memory is tight during generation, drop
  this and use rule-based + template traces instead.

**Framework: Apple MLX / mlx-lm.**
- MLX is unified-memory-native on Apple Silicon, supports efficient LoRA
  training and generation, and is the right enabler here (vs PyTorch MPS).
- Use `mlx-lm` for LoRA training + generation; custom code for envs, opponents,
  rewards, and the GRPO orchestration loop.
- Quantization: 4-bit for the mentor; the training model stays fp16/bf16 for
  stable LoRA gradients (LoRA params are tiny, optimizer states small).

**Memory budget (8GB unified, rough):**
- 0.5B fp16 weights: ~1 GB
- LoRA adapters + AdamW states (rank 16, small param count): <0.2 GB
- Activations w/ gradient checkpointing, batch=1, seq≤1024: ~1–2 GB
- Generation KV cache (one rollout at a time): ~0.3–0.8 GB
- OS + framework overhead: ~1.5–2 GB
- **Peak ~5–6 GB** — fits with headroom. Rollouts run **sequentially** (one
  generation at a time) so inference and training memory don't stack.

**Speed reality:** M1 Air is slow for this. Expect ~weeks for a small RL run.
This is a **proof-of-concept**, not a production run. Keep episode counts and
step counts modest (Section 9).

---

## 3. Pipeline overview (stages)

```
Stage 0  Baseline & eval harness              (no training)
Stage 1  SFT warmup on curated traces         (format + basic competence)
Stage 2  RL — primitives: ToM + simple games  (rule-based opponents)
Stage 3  RL — social deduction + negotiation  (rule-based opponents)
Stage 4  RL — bluffing + auctions             (rule-based opponents)
Stage 5  RL — mixed-game consolidation        (rule-based opponent mix)
Stage 6  Transfer RL on math/logic            (efficiency-aware)
Stage 7  Evaluation & ablation analysis
```

Same GRPO trainer + reward machinery throughout; only env mix, opponent
strengths, and reward-schedule weights change per stage. No LLM opponent pool
(rule-based only) — this is the main simplification vs the big-compute design.

---

## 4. Society-games corpus (Mac-feasible subset)

Full universe in `GAMES_CATALOG.md` (64 games). The **Mac-feasible training
subset** prioritizes short episodes + trivial/easy verifiers (max rollout
throughput) and drops long-horizon games (Diplomacy, Blood on the Clocktower,
Catan trade) that are too slow on-device.

**Training subset (~24 games):**
- ToM probes: 45 false-belief, 46 second-order, 47 faux pas, 48 hinting,
  49 strange stories (Phase 2)
- Repeated/coordination: 35 IPD, 37 stag hunt, 38 tit-for-tat tourney,
  51 pure coordination, 52 BoS, 53 chicken (Phase 2)
- Signaling: 30 Lewis, 31 cheap-talk, 33 Taboo (Phase 3)
- Negotiation: 11 split-the-pie, 12 ultimatum, 14 Nash bargaining (Phase 3)
- Deduction: 1 Mafia, 3 Avalon, 7 Spyfall, 8 Coup, 9 Love Letter (Phase 3)
- Bluffing: 41 Kuhn poker, 42 Leduc poker, 40 Liar's Dice (Phase 4)
- Auctions: 18 sealed first-price, 19 Vickrey, 24 all-pay (Phase 4)
- Logic duels: 61 Mastermind, 62 Bulls & Cows (Phase 5)

**Held-out (eval only, never trained):** 4 Secret Hitler, 34 Codenames,
55 Beauty Contest (text-native, exact verifiers, distinct primitives).

**Dropped for Mac run (too slow / long-horizon):** 2, 5, 6, 10, 15, 16, 17,
20–23, 25–29, 39, 43, 44, 50, 54, 56–60, 63, 64. Reinstatable on bigger
compute (Section 15).

---

## 5. Environments framework

Unified text-based env interface (PettingZoo-shaped, text-wrapped):

```
EnvSpec:
  reset() -> observation_text, legal_actions[]
  step(action_text) -> observation_text, legal_actions[], reward, done, info
  parse_action(text) -> structured_move | Invalid
  verify(structured_move) -> Legal | Illegal
  outcome() -> win/loss/score per agent
```

Per-game modules: state serializer, legal-action generator, action parser
(paraphrase-tolerant via regex + tiny LLM-judge fallback *only if needed* —
prefer strict regex to keep it cheap), exact verifier.

**Opponents: rule-based only.** Three strengths per game:
- `random` — uniform legal moves.
- `greedy` — 1-ply myopic optimum.
- `heuristic` — hand-written near-optimal bot (e.g., Nash-mixed for Kuhn
  poker, truthful for Vickrey, belief-tracking for Avalon).

**Opponent curriculum:** strength sampled so target win-rate ≈ 0.7–0.8. Start
random → greedy → heuristic as the policy's Elo rises. No snapshot pool
(self-play) on Mac — too slow to be worth it at this scale; rule-based gives a
stable, cheap ladder.

---

## 6. Task sets (transfer probes + sanity)

**Math (primary transfer target), use subsets to keep eval cheap:**
- GSM8K — 300-item stratified subset (dense, primary)
- MATH — 200-item subset (competition)
- AIME — a handful as a stretch probe (likely near-0 at 0.5B; report anyway)

**Logic (secondary):** LogiQA subset, ProofWriter subset (formal, cheap to
verify exactly — good efficiency signal).

**Reasoning-efficiency probes (custom, the novel metric):** a fixed 200-item
set from GSM8K+MATH scored for **tokens-to-correct** and **self-reversal
rate**, run pre/post each stage.

**Sanity / regression:** ARC subset, HellaSwag subset — ensure no catastrophic
forgetting / no degeneration into terse-wrong.

**Generalization:** held-out games (Secret Hitler, Codenames, Beauty Contest)
win-rate vs fixed rule-based opponents.

---

## 7. Data curation strategy (Mac-feasible)

**Goal of SFT warmup:** teach the *format* (structured reasoning then a
decisive move) + basic game competence + R1-style reasoning trace shape, so RL
doesn't waste steps learning legal moves.

**Trace sources (in priority order):**
1. **Rule-based + templates:** for most games, synthesize concise decisive
   traces by templating the heuristic bot's *internal state* into natural
   language ("I believe player 2 is the spy because they defected on mission
   1; I vote to reject."). Cheap, exact, unlimited.
2. **Mentor model (1.5B Q4, offline, sparingly):** generate a few hundred
   high-quality traces for the harder games (Avalon, Kuhn poker) where
   templates are weak. Filter strictly.
3. **(Skip if mentor too slow.)**

**Filter pipeline:** legality check → outcome filter (win or top-quartile
score) → **consistency filter** (low self-reversal marker count — this is the
whole point) → dedup by semantic similarity → family/role balance → per-game
volume cap.

**Contrast traces (high value, cheap):** also synthesize *verbose hedging*
traces that reach the same right answer, used as **contrast pairs** during
SFT/RL to teach "same outcome, fewer tokens = better". Directly operationalizes
the efficiency hypothesis at the data level.

**Volume target:** ~10–30k high-quality episodes (Mac scale). Quality and
balance dominate over volume.

---

## 8. RL policy & reward design

**Algorithm: GRPO** (group relative policy optimization; no value model — ideal
for tiny models). Group size G=4–8 rollouts per prompt (sequential on Mac).
Advantage = normalized group outcome. KL to the SFT-warmup checkpoint (keeps
format + prevents drift). LoRA updates only.

**Reward composition:**

```
R = α·R_outcome + β·R_process + γ·R_efficiency + δ·R_consistency
```

- **R_outcome** — win / normalized score in [0,1]. Primary.
- **R_process** — per-move quality vs the heuristic bot's choice, where cheap
  to compute (Kuhn/Leduc Nash value, Vickrey truthfulness, Mastermind optimal
  guess). Sparse; cuts "win by luck" noise.
- **R_efficiency** — `-λ · token_count`, **gated on correctness** (applied only
  on win/correct). Never rewards terse-wrong. λ curriculum-scheduled (0 early,
  ramped Phase 4–6).
- **R_consistency** — penalty for **self-reversals** ("wait", "actually",
  "but", "hmm", "no," followed by a reversal). Detected by a regex/heuristic
  baseline (cheap, no extra model). Directly targets "oh wait but...".

**Reward schedule by phase (illustrative):**

| Phase | α | β | γ | δ | KL |
|-------|---|---|---|---|----|
| 2 primitives | 1.0 | 0.3 | 0.0 | 0.1 | low |
| 3 deduction/neg | 1.0 | 0.2 | 0.0 | 0.2 | low |
| 4 bluff/auction | 1.0 | 0.2 | 0.1 | 0.3 | med |
| 5 consolidation | 1.0 | 0.1 | 0.2 | 0.3 | med |
| 6 math transfer | 1.0 (correct) | 0.0 | 0.3 | 0.3 | high |

**Why gate efficiency on correctness:** eliminates the classic length-penalty
failure (terse-wrong) and exactly encodes "concise *because* confident+right".

---

## 9. Curriculum & scale

**Phase sequence** (by primitive difficulty + horizon):
- **Phase 2 — Primitives:** ToM probes, signaling, PD/coordination. Short, 2
  agents. Builds belief modeling, commitment, reading intent.
- **Phase 3 — Social deduction & negotiation:** Avalon, Spyfall, Coup, Mafia,
  Love Letter, Ultimatum, Split-the-Pie, Nash bargaining, Taboo. Multi-turn,
  hidden roles.
- **Phase 4 — Bluffing & auctions:** Kuhn/Leduc poker, Liar's Dice, sealed/
  Vickrey/all-pay auctions. Mixed strategies. Efficiency reward on.
- **Phase 5 — Consolidation:** mixed rule-based opponent mix across all
  training games, full reward. Habit crystallizes.
- **Phase 6 — Transfer:** RL on math/logic with the **same** efficiency +
  consistency rewards. The test: does the habit transfer across domains?

**Inter-stage gates:** advance only when win-rate vs phase opponents ≥0.7 and
no >5% regression on prior-phase games.

**Scale (Mac reality):** a run = a few hundred to a few thousand GRPO steps
total, over days–weeks. Keep per-phase step budgets small (e.g., 200–800
steps/phase). This is enough to see directional effects, not to converge.

---

## 10. Evaluation & metrics

**Per-game:** win-rate per family vs rule-based opponents; moves-to-win.

**Math/logic (transfer):**
- Accuracy: GSM8K-subset, MATH-subset, LogiQA, ProofWriter.
- **Token efficiency:** mean tokens per *correct* answer; pre vs post.
- **Self-reversal rate:** reversal-marker frequency per correct trace.
- **Transfer index:** Δ(efficiency) of [full pipeline] minus [math-only-RL
  baseline] — the headline number for the hypothesis.

**Generalization:** held-out games win-rate.

**Regression:** ARC/HellaSwag subsets for forgetting.

**Efficiency-vs-accuracy frontier:** plot accuracy vs token-budget; success =
Pareto improvement (same accuracy, fewer tokens), ideally strict domination.

All metrics at every stage checkpoint → learning curves.

---

## 11. Ablations

| ID | Variant | Tests |
|----|---------|-------|
| A1 | Math RL only (no society games) | baseline for transfer index |
| A2 | Society games, γ=0 (no efficiency reward) | does efficiency emerge or need a push? |
| A3 | Society games + efficiency, no Phase 6 | does transfer need the bridge phase? |
| A4 | Random game mix vs curriculum | curriculum value |
| A5 | δ=0 (no consistency reward) | is explicit anti-backtrack needed? |
| A6 | No SFT warmup | cold-start value |
| A7 | γ sweep {0, 0.1, 0.2, 0.3} | efficiency/accuracy trade-off |

A1 vs full pipeline = core experiment. A2/A5 isolate the two novel rewards.
On Mac, run ablations as *short* variants (fewer steps) — directional signal
only.

---

## 12. Risks & mitigations

- **Reward hacking (terse-wrong):** efficiency gated on correctness;
  consistency reward; regression suite.
- **Too slow to be useful:** Mac reality. Mitigation: short episodes only,
  rule-based opponents, small subsets, MLX, sequential rollouts, modest step
  budgets. Accept PoC scope.
- **No transfer to math:** biggest risk. Mitigations: shared reasoning format
  across games and math; Phase 6 bridge; ablation A3 quantifies. If null at
  0.5B, **inconclusive** — note explicitly, don't overclaim.
- **Rule-based opponents too weak → policy games the bot, not reasoning:**
  Mitigation: heuristic opponents where possible (Nash-mixed poker, truthful
  auctions), held-out games for generalization, consistency reward that
  penalizes weird traces.
- **Backtrack detector noise (regex):** validate on a small human-labeled set;
  report sensitivity; keep heuristic conservative.
- **Memorization:** held-out games; family diversity; per-game volume caps.
- **Forgetting base reasoning:** KL to SFT checkpoint; regression suite;
  inter-stage gates.

---

## 13. Proposed repo structure (scaffold for later)

```
rl/
  DESIGN.md
  GAMES_CATALOG.md
  src/
    envs/         # one module per game family; unified EnvSpec
    opponents/    # random / greedy / heuristic (rule-based only on Mac)
    verifiers/    # legality + outcome per game
    rewards/      # outcome, process, efficiency, consistency
    data/         # template + mentor trace gen, filtering, contrast pairs
    train/        # MLX LoRA GRPO loop, curriculum controller
    eval/         # metrics, benchmarks, frontier plots
  configs/        # one YAML per phase (reward schedule, game mix, opponents)
  scripts/        # run_train, run_eval, gen_traces
  data/
    traces/       # raw traces
    curated/      # filtered SFT set
    checkpoints/  # LoRA adapters
    eval_out/
```

---

## 14. Open decisions (to confirm before implementation)

1. **Training base:** `Qwen2.5-0.5B-Instruct` (recommended) vs trying
   `DeepSeek-R1-Distill-Qwen-1.5B` LoRA at the OOM edge (risky, very slow).
2. **Mentor traces:** use the 1.5B Q4 mentor offline for a few hundred traces,
   or go template-only (cheaper, fully Mac-native)?
3. **Backtrack detector:** regex-only (cheap) vs regex + tiny trained
   classifier (needs a labeled set).
4. **First runnable scope:** full 7-stage PoC, or a minimal
   Phase-2 → Phase-6 vertical slice (a few games + math transfer) to test the
   hypothesis cheapest? (Recommended: vertical slice first.)

---

## 15. Appendix — Scaling to multi-GPU (original larger design)

When cloud/multi-GPU compute is available, lift these constraints:
- **Base:** `DeepSeek-R1-Distill-Qwen-1.5B`, full-FT (not LoRA).
- **Opponents:** add an **LLM mentor pool** (7B–14B) + **snapshot-pool
  fictitious play** for self-play, sampled to ~80% win-rate. Drop the
  rule-based-only restriction.
- **Games:** reinstate the full 64-game catalog incl. long-horizon (Diplomacy,
  Blood on the Clocktower, Catan trade, combinatorial auctions).
- **Data:** ~50–150k expert traces from a frontier reasoning model, strictly
  filtered; contrast pairs retained.
- **Scale:** full convergence runs; all 8 ablations at full step budgets.
- **Method:** GRPO (or DAPO) with KL to base; same 4-term reward, same
  curriculum, same eval. The Mac design is a strict subset of this, so results
  transfer upward.
- **Everything else (hypothesis, reward design, curriculum logic, eval
  metrics, ablation structure) is identical** — the Mac run is a small-scale
  probe of the same mechanism.
