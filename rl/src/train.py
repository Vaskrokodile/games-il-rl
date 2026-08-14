"""Main entry point for society-game GRPO training.

Usage:
    python -m rl.src.train --steps 100
    python -m rl.src.train --steps 100 --phase 2
    python -m rl.src.train --steps 10 --dry-run  # quick test
"""
import argparse
import sys
import os

# Add parent dir to path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grpo import GRPOTrainer


# ── Phase configs ───────────────────────────────────────────────────────────

PHASES = {
    2: {
        "name": "primitives",
        "games": {
            "false_belief": 1.5,       # ToM probes
            "prisoners_dilemma": 1.0,  # cooperation/defection
            "coordination": 1.0,       # equilibrium selection
        },
        "reward": {
            "alpha": 1.0, "beta": 0.3, "gamma": 0.0, "delta": 0.1,
            "efficiency_lambda": 0.001,
        },
        "lr": 1e-5,
        "group_size": 2,   # 2 for 1.5B on 8GB, 4 for 0.5B
        "max_tokens": 80,  # short for memory safety on 1.5B
    },
    3: {
        "name": "deduction_negotiation",
        "games": {
            "ultimatum": 1.0,
            "prisoners_dilemma": 0.5,
            "coordination": 0.5,
            "false_belief": 0.5,
            "sealed_bid_auction": 1.0,
        },
        "reward": {
            "alpha": 1.0, "beta": 0.2, "gamma": 0.0, "delta": 0.2,
            "efficiency_lambda": 0.001,
        },
        "lr": 1e-5,
        "group_size": 2,
        "max_tokens": 80,
    },
    4: {
        "name": "bluffing_auctions",
        "games": {
            "sealed_bid_auction": 1.5,
            "ultimatum": 1.0,
            "prisoners_dilemma": 0.3,
        },
        "reward": {
            "alpha": 1.0, "beta": 0.2, "gamma": 0.1, "delta": 0.3,
            "efficiency_lambda": 0.001,
        },
        "lr": 5e-6,
        "group_size": 2,
        "max_tokens": 80,
    },
    6: {
        "name": "math_transfer",
        "games": {
            "false_belief": 0.3,
            "coordination": 0.3,
            "ultimatum": 0.3,
        },
        "reward": {
            "alpha": 1.0, "beta": 0.0, "gamma": 0.3, "delta": 0.3,
            "efficiency_lambda": 0.002,
        },
        "lr": 5e-6,
        "group_size": 2,
        "max_tokens": 80,
    },
}


def main():
    parser = argparse.ArgumentParser(description="Society-game GRPO training")
    parser.add_argument("--steps", type=int, default=100, help="Number of training steps")
    parser.add_argument("--phase", type=int, default=2, choices=[2, 3, 4, 6],
                        help="Curriculum phase")
    parser.add_argument("--model", type=str, default="auryn-macmillan/boostedv1-ilv9",
                        help="Model path or HF repo")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to existing LoRA adapters to resume from")
    parser.add_argument("--lora-rank", type=int, default=4, help="LoRA rank (4 for 1.5B, 8 for 0.5B)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--group-size", type=int, default=None, help="Override group size")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override max generation tokens")
    parser.add_argument("--save-every", type=int, default=50, help="Save checkpoint every N steps")
    parser.add_argument("--dry-run", action="store_true", help="Quick 3-step test run")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints")
    parser.add_argument("--log-dir", type=str, default="data/logs")
    args = parser.parse_args()

    phase_cfg = PHASES[args.phase]
    print(f"\nPhase {args.phase}: {phase_cfg['name']}")

    if args.dry_run:
        args.steps = 3
        args.save_every = 3
        print("*** DRY RUN: 3 steps only ***")

    lora_config = {"rank": args.lora_rank, "alpha": args.lora_rank * 2, "dropout": 0.0, "scale": 10.0}

    trainer = GRPOTrainer(
        model_path=args.model,
        adapter_path=args.adapter,
        lora_config=lora_config,
        games_config=phase_cfg["games"],
        reward_config=phase_cfg["reward"],
        learning_rate=args.lr or phase_cfg["lr"],
        group_size=args.group_size or phase_cfg["group_size"],
        max_tokens=args.max_tokens or phase_cfg["max_tokens"],
        checkpoint_dir=os.path.join(os.path.dirname(__file__), "..", args.checkpoint_dir),
        log_dir=os.path.join(os.path.dirname(__file__), "..", args.log_dir),
    )

    trainer.train(
        num_steps=args.steps,
        save_every=args.save_every,
        log_every=1,
    )


if __name__ == "__main__":
    main()
