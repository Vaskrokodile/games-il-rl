# Society-Games Catalog

> Companion to `DESIGN.md`. A curated, abundant taxonomy of games/environments
> for training decisive, efficient multi-agent reasoning on a 1.5B model.
>
> Columns:
> - **Text-native**: can state + act in text only (no vision). Y/N.
> - **Verifier**: how hard to compute legality + outcome exactly.
>   `trivial` / `easy` / `medium` / `hard`.
> - **Reward**: `win` (binary), `score` (continuous), `payoff` (game-theoretic).
> - **Phase**: curriculum phase(s) where it's useful (2 primitives, 3 deduction,
>   4 deep, 5 consolidation). `H` = held-out (eval only, never trained).
> - **Primitive**: the main reasoning skill stressed.
>
> Held-out games (~15%) are marked **H** and excluded from training.
>
> **Mac-feasible subset:** see `DESIGN.md` Section 4 for the ~24-game training
> subset used in the 8GB M1 Air run (short episodes + trivial/easy verifiers
> only). Long-horizon games (Diplomacy, Blood on the Clocktower, Catan trade,
> combinatorial auctions, etc.) are dropped on Mac and reinstated at scale
> (DESIGN.md Section 15).

---

## A. Social deduction

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 1 | Mafia (classic) | Y | easy | win | 3 | deception/credibility |
| 2 | One Night Ultimate Werewolf | Y | easy | win | 3 | belief update under claims |
| 3 | The Resistance / Avalon | Y | easy | win | 3 | sustained lie detection |
| 4 | Secret Hitler | Y | medium | win | **H** | coalition + deduction |
| 5 | Blood on the Clocktower (roles subset) | Y | medium | win | 5 | role inference |
| 6 | Two Rooms and a Boom | Y | easy | win | 3 | team coordination w/ hidden info |
| 7 | Spyfall | Y | easy | win | 3 | question design / evasion |
| 8 | Coup | Y | easy | win | 3 | bluff under challenge |
| 9 | Love Letter (deduction) | Y | easy | win | 3 | card-counting + bluff |
| 10 | Don't Get Got (micro-deduction) | Y | trivial | win | 2 | light ToM |

## B. Negotiation & bargaining

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 11 | Split-the-Pie (Anthropic) | Y | trivial | payoff | 3 | fair division + leverage |
| 12 | Ultimatum game | Y | trivial | payoff | 2 | fairness + threat credibility |
| 13 | Dictator (as recipient, predict) | Y | trivial | payoff | 2 | preference modeling |
| 14 | Nash bargaining (text) | Y | easy | payoff | 3 | surplus splitting |
| 15 | Deal or No Deal (text) | Y | easy | payoff | 3 | risk + expected value |
| 16 | Catan-style trading (simplified) | Y | medium | score | 4 | multi-party trade + trust |
| 17 | Monopoly trade (simplified) | Y | medium | payoff | 4 | valuation + leverage |

## C. Auctions & markets

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 18 | Sealed-bid first price | Y | trivial | payoff | 4 | shading + value estimation |
| 19 | Vickrey (second-price) | Y | trivial | payoff | 4 | truthfulness incentives |
| 20 | English ascending | Y | easy | payoff | 4 | dropout timing |
| 21 | Dutch descending | Y | easy | payoff | 4 | jump-in risk |
| 22 | Double auction (market making) | Y | medium | payoff | 4 | price discovery |
| 23 | Combinatorial/spectrum (simplified) | Y | medium | payoff | 5 | exposure + package bidding |
| 24 | All-pay auction | Y | trivial | payoff | 4 | contest theory |

## D. Coalition & diplomacy

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 25 | Simplified Diplomacy (2–4 powers) | Y | medium | score | 4 | long-horizon alliance + betrayal |
| 26 | Weighted voting / coalition formation | Y | easy | payoff | 4 | power indices |
| 27 | Hedonic games (room assignment) | Y | easy | payoff | 5 | stability notions |
| 28 | Stable matching (deferred acceptance) | Y | easy | payoff | 4 | mechanism + strategy |
| 29 | Strategic voting (plurality/Borda/STV) | Y | easy | payoff | 4 | manipulation under rules |

## E. Signaling & communication

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 30 | Lewis signaling game | Y | trivial | win | 2 | convention emergence |
| 31 | Cheap-talk coordination | Y | trivial | win | 2 | credible communication |
| 32 | Code-word cooperation | Y | easy | win | 3 | shared code under pressure |
| 33 | Taboo (describe w/o forbidden words) | Y | easy | win | 3 | efficient expression |
| 34 | Codenames (spymaster) | Y | medium | win | **H** | graded signaling |

## F. Repeated & iterated

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 35 | Iterated Prisoner's Dilemma | Y | trivial | payoff | 2 | cooperation/defection policy |
| 36 | Iterated Snowdrift | Y | trivial | payoff | 2 | anti-coordination |
| 37 | Iterated Stag Hunt | Y | trivial | payoff | 2 | trust + payoff dominance |
| 38 | Tit-for-tat tournament | Y | trivial | payoff | 2 | strategy identification |
| 39 | Repeated trust game | Y | trivial | payoff | 3 | reputation building |

## G. Bluffing & incomplete information

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 40 | Liar's Dice (Perudo) | Y | easy | win | 4 | probabilistic bluff |
| 41 | Kuhn Poker (3-card) | Y | trivial | payoff | 4 | mixed strategies, exact solvable |
| 42 | Leduc Poker | Y | easy | payoff | 4 | larger state, mixed strategies |
| 43 | Limit Hold'em (heads-up, simplified) | Y | medium | payoff | 5 | deep bluff + pot odds |
| 44 | Bluff (card game) | Y | easy | win | 4 | challenge calibration |

## H. Theory-of-mind probes

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 45 | False-belief (Sally-Anne) | Y | trivial | win | 2 | first-order ToM |
| 46 | Second-order false belief | Y | trivial | win | 2 | nested belief |
| 47 | Faux pas recognition | Y | easy | win | 2 | social inference |
| 48 | Hinting task | Y | easy | win | 2 | indirect speech |
| 49 | Strange stories (Happé) | Y | easy | win | 2 | pragmatic reasoning |
| 50 | Text perspective-taking | Y | easy | win | 2 | belief-asymmetry |

## I. Coordination & pure strategy

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 51 | Pure coordination | Y | trivial | win | 2 | equilibrium selection |
| 52 | Battle of the Sexes | Y | trivial | payoff | 2 | anti-coordination |
| 53 | Chicken | Y | trivial | payoff | 2 | credible commitment |
| 54 | Minimum-effort (weak link) | Y | trivial | payoff | 3 | team coordination |
| 55 | Keynesian beauty contest (guess 2/3 avg) | Y | trivial | win | **H** | iterated reasoning depth |

## J. Resource & scheduling (strategic)

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 56 | Competitive job-shop scheduling | Y | medium | score | 5 | planning under contention |
| 57 | Knapsack auction | Y | easy | payoff | 4 | bidding + packing |
| 58 | Bin-packing race | Y | easy | win | 5 | fast optimization |
| 59 | Network routing (Braess) | Y | easy | payoff | 4 | congestion + routing |
| 60 | Queue/latency game | Y | easy | payoff | 4 | strategic waiting |

## K. Logic-puzzle-adjacent duels

| # | Game | Text-native | Verifier | Reward | Phase | Primitive |
|---|------|-------------|----------|--------|-------|-----------|
| 61 | Mastermind (maker + breaker) | Y | trivial | win | 3 | hypothesis testing |
| 62 | Bulls and Cows | Y | trivial | win | 3 | constraint elimination |
| 63 | Constraint-satisfaction duel | Y | medium | win | 5 | search + pruning |
| 64 | Logic-grid race | Y | medium | win | 5 | structured deduction under time |

---

## Notes on selection

- **Held-out (H):** Secret Hitler (4), Codenames (34), Beauty Contest (55) —
  chosen because they're text-native, have exact verifiers, and stress
  distinct primitives (coalition deduction, graded signaling, iterated
  reasoning depth). Good generalization probes.
- **Cheap-exact-process-reward games** (best for R_process): Kuhn/Leduc poker
  (Nash-equilibrium value computable), Vickrey/all auctions (truthfulness
  check), stable matching, Mastermind/Bulls (optimal next guess computable).
  These give the cleanest per-move signal.
- **Short-episode, high-throughput** (bulk of rollouts): everything in H, I,
  F, and B's trivial-verifier items.
- **Long-horizon stress** (fewer rollouts, high value): Diplomacy (25),
  Blood on the Clocktower (5), Catan trade (16).
- **Coverage matrix:** ensure each *primitive* column has >=3 training games so
  no single game can dominate a skill.
