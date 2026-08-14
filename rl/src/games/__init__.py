"""Society game implementations — single-turn, text-based, exact verifiers.

Each game samples a state (with opponent behavior pre-determined), presents it
as a text prompt, parses the model's action, and resolves the outcome.
"""
import random
import re
from .base import BaseGame, GameState, GameOutcome


# ── Ultimatum Game ──────────────────────────────────────────────────────────

class UltimatumGame(BaseGame):
    """Propose a split of $10. Opponent (pre-sampled) accepts or rejects.
    If rejected, both get $0. Tests fairness modeling + leverage."""
    name = "ultimatum"
    game_type = "ultimatum"

    OPPONENT_THRESHOLDS = [3, 4, 5, 6, 7]  # opponent rejects if your share < threshold

    def sample_state(self) -> GameState:
        threshold = random.choice(self.OPPONENT_THRESHOLDS)
        your_share = random.randint(1, 9)  # you keep this, opponent gets 10 - your_share
        prompt = (
            f"You are playing the Ultimatum Game. There is $10 to split between you and "
            f"another player. You propose how much you keep (1-9); the other player gets "
            f"the rest. If they reject your proposal, both get $0.\n\n"
            f"What amount do you keep? Give your reasoning, then end with: "
            f"KEEP: <amount>"
        )
        return GameState(
            prompt=prompt,
            legal_actions=[str(i) for i in range(1, 10)],
            game_type=self.game_type,
            metadata={"threshold": threshold, "pot": 10},
        )

    def parse_action(self, text: str, state: GameState) -> str | None:
        m = re.search(r"KEEP:\s*(\d+)", text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 9:
                return str(val)
        # fallback: look for last number 1-9
        nums = re.findall(r"\b([1-9])\b", text)
        return nums[-1] if nums else None

    def resolve(self, action: str, state: GameState) -> GameOutcome:
        your_share = int(action)
        opponent_share = state.metadata["pot"] - your_share
        threshold = state.metadata["threshold"]
        # opponent accepts if opponent_share >= threshold
        accepted = opponent_share >= threshold
        if accepted:
            outcome = your_share / state.metadata["pot"]  # normalize
        else:
            outcome = 0.0
        # process: optimal is to keep (10 - threshold) — maximum accepted share
        optimal = state.metadata["pot"] - threshold
        process = 1.0 - abs(your_share - optimal) / state.metadata["pot"]
        return GameOutcome(outcome=outcome, process_quality=max(0, process),
                           info={"accepted": accepted, "your_share": your_share,
                                 "opponent_share": opponent_share, "optimal": optimal})


# ── Prisoner's Dilemma ──────────────────────────────────────────────────────

class PrisonersDilemma(BaseGame):
    """Classic PD. Opponent action pre-sampled. Tests cooperation/defection."""
    name = "prisoners_dilemma"
    game_type = "pd"
    # Payoffs: (your, opponent) for (C,C), (C,D), (D,C), (D,D)
    PAYOFFS = {("C", "C"): (3, 3), ("C", "D"): (0, 5),
               ("D", "C"): (5, 0), ("D", "D"): (1, 1)}

    def sample_state(self) -> GameState:
        opp_action = random.choice(["C", "D"])
        # sometimes reveal opponent's last action (iterated context)
        reveal = random.choice([True, False])
        opp_desc = f"The other player chose {opp_action} in the last round. " if reveal else ""
        prompt = (
            f"You are playing the Prisoner's Dilemma. {opp_desc}You can Cooperate (C) "
            f"or Defect (D). Payoffs: both cooperate -> 3 each; you defect, they cooperate "
            f"-> you get 5; both defect -> 1 each; you cooperate, they defect -> you get 0.\n\n"
            f"What do you choose? Give your reasoning, then end with: CHOICE: C or CHOICE: D"
        )
        return GameState(
            prompt=prompt,
            legal_actions=["C", "D"],
            game_type=self.game_type,
            metadata={"opp_action": opp_action},
        )

    def parse_action(self, text: str, state: GameState) -> str | None:
        m = re.search(r"CHOICE:\s*([CD])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # fallback
        if re.search(r'\bdefect\b', text, re.I):
            return "D"
        if re.search(r'\bcooperate\b', text, re.I):
            return "C"
        return None

    def resolve(self, action: str, state: GameState) -> GameOutcome:
        opp = state.metadata["opp_action"]
        your_payoff, _ = self.PAYOFFS[(action, opp)]
        outcome = your_payoff / 5.0  # normalize (max payoff = 5)
        # process: against D, optimal is D (1 vs 0); against C, optimal is D (5 vs 3)
        # but we reward mutual cooperation slightly to encourage non-greedy
        if action == "C" and opp == "C":
            process = 1.0
        elif action == "D":
            process = 0.7  # defection is rational but not "pro-social"
        else:
            process = 0.2  # sucker's payoff
        return GameOutcome(outcome=outcome, process_quality=process,
                           info={"your_payoff": your_payoff, "opp_action": opp})


# ── Pure Coordination ───────────────────────────────────────────────────────

class CoordinationGame(BaseGame):
    """Choose A or B. Win if you match the opponent. Tests equilibrium selection."""
    name = "coordination"
    game_type = "coordination"

    def sample_state(self) -> GameState:
        opp_choice = random.choice(["A", "B"])
        # hint: sometimes one option has higher payoff (stag hunt variant)
        high = random.choice(["A", "B"])
        prompt = (
            f"You and another player each choose A or B. If you both choose the same, "
            f"you both win. Choosing {high} gives a bigger reward if matched.\n\n"
            f"What do you choose? Give your reasoning, then end with: CHOICE: A or CHOICE: B"
        )
        return GameState(
            prompt=prompt,
            legal_actions=["A", "B"],
            game_type=self.game_type,
            metadata={"opp_choice": opp_choice, "high": high},
        )

    def parse_action(self, text: str, state: GameState) -> str | None:
        m = re.search(r"CHOICE:\s*([AB])", text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        if re.search(r'\bA\b', text):
            return "A"
        if re.search(r'\bB\b', text):
            return "B"
        return None

    def resolve(self, action: str, state: GameState) -> GameOutcome:
        opp = state.metadata["opp_choice"]
        matched = action == opp
        if matched:
            outcome = 1.0 if action == state.metadata["high"] else 0.7
        else:
            outcome = 0.0
        process = 1.0 if matched else 0.0
        return GameOutcome(outcome=outcome, process_quality=process,
                           info={"matched": matched, "opp_choice": opp})


# ── Sealed-Bid Auction ──────────────────────────────────────────────────────

class SealedBidAuction(BaseGame):
    """Bid for an item worth V. Opponent bid pre-sampled. Tests value estimation."""
    name = "sealed_bid_auction"
    game_type = "auction"

    def sample_state(self) -> GameState:
        value = random.randint(10, 100)
        opp_bid = random.randint(0, value + 10)
        prompt = (
            f"You are in a sealed-bid first-price auction. An item is worth {value} to you. "
            f"You submit one bid. The highest bid wins and pays their bid. "
            f"If you tie, you win with 50% chance.\n\n"
            f"What is your bid (0-{value + 20})? Give your reasoning, then end with: BID: <amount>"
        )
        return GameState(
            prompt=prompt,
            legal_actions=[str(i) for i in range(0, value + 21)],
            game_type=self.game_type,
            metadata={"value": value, "opp_bid": opp_bid},
        )

    def parse_action(self, text: str, state: GameState) -> str | None:
        m = re.search(r"BID:\s*(\d+)", text, re.IGNORECASE)
        if m:
            return m.group(1)
        nums = re.findall(r"\b(\d+)\b", text)
        return nums[-1] if nums else None

    def resolve(self, action: str, state: GameState) -> GameOutcome:
        bid = int(action)
        value = state.metadata["value"]
        opp_bid = state.metadata["opp_bid"]
        if bid > opp_bid:
            surplus = value - bid
            outcome = max(0, surplus) / value if value > 0 else 0
        elif bid == opp_bid:
            surplus = (value - bid) * 0.5
            outcome = max(0, surplus) / value if value > 0 else 0
        else:
            outcome = 0.0
        # process: optimal bid is roughly opp_bid + 1 (if < value)
        optimal = min(opp_bid + 1, value - 1) if value > 1 else 0
        process = 1.0 - abs(bid - optimal) / max(value, 1)
        return GameOutcome(outcome=min(1, outcome), process_quality=max(0, process),
                           info={"bid": bid, "value": value, "opp_bid": opp_bid,
                                 "won": bid >= opp_bid})


# ── Theory of Mind: False Belief ────────────────────────────────────────────

class FalseBeliefTask(BaseGame):
    """Sally-Anne style false belief test. Tests first-order ToM."""
    name = "false_belief"
    game_type = "tom"

    SCENARIOS = [
        {"story": "Sally puts a ball in a basket and leaves. Anne moves the ball "
                  "to a box. Sally returns. Where will Sally look for the ball?",
         "answer": "basket", "distractors": ["box"]},
        {"story": "John leaves his keys on the table. His wife moves them to the "
                  "drawer. When John comes home, where will he look for his keys?",
         "answer": "table", "distractors": ["drawer"]},
        {"story": "Mary puts cookies in the cookie jar. Tom eats them and puts "
                  "the jar back empty. When Mary wants a cookie, where will she look?",
         "answer": "cookie jar", "distractors": ["empty", "nowhere"]},
        {"story": "A boy hides a toy under his bed. His sister moves it to the "
                  "closet while he's away. Where will the boy look first?",
         "answer": "bed", "distractors": ["closet"]},
    ]

    def sample_state(self) -> GameState:
        s = random.choice(self.SCENARIOS)
        choices = [s["answer"]] + s["distractors"]
        random.shuffle(choices)
        prompt = (
            f"{s['story']}\n\n"
            f"Options: {', '.join(choices)}\n\n"
            f"Give your reasoning, then end with: ANSWER: <option>"
        )
        return GameState(
            prompt=prompt,
            legal_actions=choices,
            game_type=self.game_type,
            metadata={"answer": s["answer"], "choices": choices},
        )

    def parse_action(self, text: str, state: GameState) -> str | None:
        m = re.search(r"ANSWER:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
        if m:
            ans = m.group(1).strip().lower()
            for c in state.legal_actions:
                if c.lower() in ans or ans in c.lower():
                    return c
        # fallback: check if any choice appears in text
        lower = text.lower()
        for c in state.legal_actions:
            if c.lower() in lower:
                return c
        return None

    def resolve(self, action: str, state: GameState) -> GameOutcome:
        correct = action == state.metadata["answer"]
        return GameOutcome(
            outcome=1.0 if correct else 0.0,
            process_quality=1.0 if correct else 0.0,
            info={"correct": correct, "answer": state.metadata["answer"]},
        )


# ── Game Registry ───────────────────────────────────────────────────────────

GAMES = {
    "ultimatum": UltimatumGame,
    "prisoners_dilemma": PrisonersDilemma,
    "coordination": CoordinationGame,
    "sealed_bid_auction": SealedBidAuction,
    "false_belief": FalseBeliefTask,
}
