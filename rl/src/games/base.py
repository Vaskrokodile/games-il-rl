"""Base environment interface for society games.

All games are single-turn (one observation -> one reasoning+action -> outcome)
to keep rollout cost low and memory predictable on 8GB M1 Air.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GameState:
    """A single game instance ready for the model to act on."""
    prompt: str           # text description of the situation
    legal_actions: list   # list of valid action strings
    game_type: str        # e.g. "ultimatum", "pd", "coordination"
    metadata: dict = field(default_factory=dict)  # game-specific info for reward


@dataclass
class GameOutcome:
    """Result of a completed game."""
    outcome: float        # normalized [0, 1] — 1=win, 0=loss, fractional for scores
    process_quality: float  # [0, 1] — how close to optimal (0 if no heuristic)
    info: dict = field(default_factory=dict)


class BaseGame:
    """Base class for a society game environment."""
    name: str = "base"
    game_type: str = "base"

    def sample_state(self) -> GameState:
        """Sample a random game instance."""
        raise NotImplementedError

    def parse_action(self, text: str, state: GameState) -> Optional[str]:
        """Parse the model's generated text into a structured action.
        Returns None if no valid action found."""
        raise NotImplementedError

    def resolve(self, action: str, state: GameState) -> GameOutcome:
        """Resolve the game given the model's action and the pre-sampled state.
        The opponent's behavior is baked into the state (pre-sampled)."""
        raise NotImplementedError
