"""Reward computation: outcome + process + efficiency(gated) + consistency.

R = α·R_outcome + β·R_process + γ·R_efficiency + δ·R_consistency

- R_efficiency = -λ · token_count, GATED on correctness (only when outcome > 0)
- R_consistency = penalty for self-reversal markers in the reasoning trace
"""
import re

# Self-reversal markers — the "oh wait but..." behavior we want to reduce
REVERSAL_PATTERNS = [
    r"\bwait\b",
    r"\bactually[,.]?",
    r"\bhmm\b",
    r"\boh\s+but\b",
    r"\bno[,.]\s",
    r"\bnevermind\b",
    r"\bsorry[,.]\s",
    r"\bcorrection\b",
    r"\bI\s+was\s+wrong\b",
    r"\blet\s+me\s+rethink\b",
    r"\bon\s+second\s+thought\b",
]
_REVERSAL_RE = re.compile("|".join(REVERSAL_PATTERNS), re.IGNORECASE)


def count_reversals(text: str) -> int:
    """Count self-reversal markers in generated text."""
    return len(_REVERSAL_RE.findall(text))


def compute_reward(
    outcome: float,
    process_quality: float,
    generated_text: str,
    token_count: int,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.0,
    delta: float = 0.1,
    efficiency_lambda: float = 0.001,
) -> dict:
    """Compute the composite reward and its components.

    Returns dict with total reward and per-component breakdown for logging.
    """
    r_outcome = alpha * outcome
    r_process = beta * process_quality

    # Efficiency: gated on correctness (outcome > 0 means win/correct)
    is_correct = outcome > 0
    if is_correct and gamma > 0:
        r_efficiency = gamma * (-efficiency_lambda * token_count)
    else:
        r_efficiency = 0.0

    # Consistency: penalize reversals (always applied if delta > 0)
    n_reversals = count_reversals(generated_text)
    r_consistency = delta * (-0.1 * n_reversals)

    total = r_outcome + r_process + r_efficiency + r_consistency

    return {
        "total": total,
        "outcome": r_outcome,
        "process": r_process,
        "efficiency": r_efficiency,
        "consistency": r_consistency,
        "n_reversals": n_reversals,
        "token_count": token_count,
        "is_correct": is_correct,
    }
