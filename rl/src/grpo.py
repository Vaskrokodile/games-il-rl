"""GRPO (Group Relative Policy Optimization) training loop.

For each step:
1. Sample a game state
2. Generate G rollouts (completions) with the current policy
3. Parse actions, resolve outcomes, compute rewards
4. Compute group-relative advantages: A_i = (R_i - mean) / (std + eps)
5. Compute logprobs of generated tokens (with gradients)
6. Loss = -mean(logprob * advantage) — policy gradient
7. Backprop and update LoRA params

Memory-safe: sequential generation, gradient checkpointing, RAM check per step.
"""
import time
import json
import os
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_map
from typing import Optional

from .model_utils import ModelWrapper
from .games import GAMES
from .rewards import compute_reward
from .memory_guard import check, MemoryGuardError, reset_peak


class GRPOTrainer:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-0.5B-Instruct",
        adapter_path: Optional[str] = None,
        lora_config: Optional[dict] = None,
        games_config: Optional[dict] = None,
        reward_config: Optional[dict] = None,
        learning_rate: float = 1e-5,
        group_size: int = 4,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        max_seq_length: int = 512,
        checkpoint_dir: str = "data/checkpoints",
        log_dir: str = "data/logs",
    ):
        # Model
        self.model_wrapper = ModelWrapper(model_path, adapter_path, lora_config)
        self.model = self.model_wrapper.model

        # Optimizer (AdamW for LoRA params)
        self.optimizer = optim.AdamW(learning_rate=learning_rate, betas=(0.9, 0.95))
        self.lr = learning_rate

        # GRPO config
        self.group_size = group_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_seq_length = max_seq_length

        # Games
        self.games_config = games_config or {
            "ultimatum": 1.0, "prisoners_dilemma": 1.0,
            "coordination": 1.0, "sealed_bid_auction": 1.0, "false_belief": 1.0,
        }
        self._game_instances = {name: cls() for name, cls in GAMES.items()}
        self._game_names = list(self.games_config.keys())
        self._game_weights = np.array(
            [self.games_config.get(n, 0) for n in self._game_names], dtype=float
        )
        self._game_weights /= self._game_weights.sum()

        # Reward config
        self.reward_config = reward_config or {
            "alpha": 1.0, "beta": 0.3, "gamma": 0.0, "delta": 0.1,
            "efficiency_lambda": 0.001,
        }

        # Logging
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        self.step = 0
        self.log_history = []

    def _sample_game(self):
        idx = np.random.choice(len(self._game_names), p=self._game_weights)
        return self._game_names[idx], self._game_instances[self._game_names[idx]]

    def _rollout(self, game_name, game):
        """Generate one rollout: sample state, generate, parse, resolve, reward."""
        state = game.sample_state()
        result = self.model_wrapper.generate(
            state.prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )

        # Truncate if too long (memory safety)
        if len(result["prompt_ids"]) + len(result["generated_ids"]) > self.max_seq_length:
            result["generated_ids"] = result["generated_ids"][:self.max_seq_length - len(result["prompt_ids"])]

        action = game.parse_action(result["text"], state)
        if action is None:
            # Invalid action penalty
            outcome_obj = type(game.resolve("", state))(
                outcome=0.0, process_quality=0.0,
                info={"invalid": True, "text": result["text"][:200]}
            )
        else:
            outcome_obj = game.resolve(action, state)

        reward_info = compute_reward(
            outcome=outcome_obj.outcome,
            process_quality=outcome_obj.process_quality,
            generated_text=result["text"],
            token_count=result["token_count"],
            **self.reward_config,
        )

        return {
            "game": game_name,
            "state": state,
            "generated_text": result["text"],
            "generated_ids": result["generated_ids"],
            "prompt_ids": result["prompt_ids"],
            "action": action,
            "outcome": outcome_obj.outcome,
            "reward": reward_info["total"],
            "reward_breakdown": reward_info,
            "valid": action is not None,
        }

    def train_step(self):
        """One GRPO step: sample game, generate G rollouts, compute advantages, update."""
        self.step += 1
        t0 = time.time()

        # 1. Sample game
        game_name, game = self._sample_game()

        # 2. Generate G rollouts (sequential)
        rollouts = []
        for g in range(self.group_size):
            try:
                check()
                r = self._rollout(game_name, game)
                rollouts.append(r)
            except MemoryGuardError as e:
                print(f"[grpo] Memory guard triggered at rollout {g}: {e}")
                # Save and stop
                self.save_checkpoint()
                raise

        if len(rollouts) < 2:
            print(f"[grpo] Only {len(rollouts)} valid rollouts, skipping step")
            return None

        # 3. Compute group-relative advantages
        rewards = np.array([r["reward"] for r in rollouts], dtype=np.float32)
        mean_r = rewards.mean()
        std_r = rewards.std()
        advantages = (rewards - mean_r) / (std_r + 1e-8)

        # 4. Compute loss and gradients using nn.value_and_grad
        self.model_wrapper.train_mode()

        # Prepare rollout data for the loss function
        rollout_data = [
            (r["prompt_ids"], r["generated_ids"], float(advantages[i]))
            for i, r in enumerate(rollouts) if r["generated_ids"]
        ]
        n_tokens_total = sum(len(gen) for _, gen, _ in rollout_data)

        if n_tokens_total == 0:
            print("[grpo] No tokens to train on, skipping")
            return None

        model = self.model

        def loss_fn(m, data):
            """GRPO policy gradient loss: -mean(logprob * advantage)."""
            total = mx.array(0.0)
            for prompt_ids, gen_ids, adv in data:
                full_ids = prompt_ids + gen_ids
                input_ids = mx.array([full_ids])
                logits = m(input_ids)[0]  # [seq_len, vocab]
                gen_start = len(prompt_ids)
                pred_logits = logits[gen_start - 1 : gen_start - 1 + len(gen_ids)]
                log_probs = nn.log_softmax(pred_logits, axis=-1)
                token_log_probs = log_probs[mx.arange(len(gen_ids)), mx.array(gen_ids)]
                total = total - (token_log_probs * adv).sum()
            return total / len(data)

        check()
        loss_and_grad = nn.value_and_grad(model, loss_fn)
        loss_val, grads = loss_and_grad(model, rollout_data)

        # Gradient clipping
        grad_flat = tree_flatten(grads)
        grad_norm = mx.sqrt(sum(mx.sum(g * g) for _, g in grad_flat))
        max_norm = 1.0
        scale = mx.minimum(1.0, max_norm / (grad_norm + 1e-6))
        grads = tree_map(lambda g: g * scale, grads)

        self.optimizer.update(model, grads)
        mx.eval(model.parameters())

        # 6. Cleanup
        mx.clear_cache()
        self.model_wrapper.eval_mode()

        # 7. Log
        elapsed = time.time() - t0
        log_entry = {
            "step": self.step,
            "game": game_name,
            "elapsed_s": round(elapsed, 2),
            "mean_reward": round(float(mean_r), 4),
            "std_reward": round(float(std_r), 4),
            "mean_outcome": round(float(np.mean([r["outcome"] for r in rollouts])), 4),
            "valid_rate": round(float(np.mean([r["valid"] for r in rollouts])), 4),
            "mean_tokens": round(float(np.mean([r["reward_breakdown"]["token_count"] for r in rollouts])), 1),
            "mean_reversals": round(float(np.mean([r["reward_breakdown"]["n_reversals"] for r in rollouts])), 2),
            "grad_norm": round(float(grad_norm), 4),
            "loss": round(float(loss_val), 4),
        }
        self.log_history.append(log_entry)

        return log_entry

    def save_checkpoint(self):
        path = os.path.join(self.checkpoint_dir, f"adapters_step{self.step}.safetensors")
        try:
            self.model_wrapper.save_adapters(path)
            # Also save log
            log_path = os.path.join(self.log_dir, f"train_log_step{self.step}.json")
            with open(log_path, "w") as f:
                json.dump(self.log_history, f, indent=2)
            print(f"[grpo] Checkpoint saved: {path}")
        except Exception as e:
            print(f"[grpo] Failed to save checkpoint: {e}")

    def train(self, num_steps: int, save_every: int = 50, log_every: int = 1):
        """Run GRPO training for num_steps."""
        print(f"\n{'='*60}")
        print(f"GRPO Training: {num_steps} steps, group_size={self.group_size}")
        print(f"Games: {self._game_names}")
        print(f"Reward: {self.reward_config}")
        print(f"LR: {self.lr}, Max tokens: {self.max_tokens}")
        print(f"{'='*60}\n")

        reset_peak()

        try:
            for step in range(num_steps):
                try:
                    log = self.train_step()
                    if log and (step % log_every == 0):
                        mem = check()
                        print(
                            f"step {log['step']:4d} | {log['game']:20s} | "
                            f"R={log['mean_reward']:+.3f}±{log['std_reward']:.3f} | "
                            f"outcome={log['mean_outcome']:.3f} | "
                            f"valid={log['valid_rate']:.2f} | "
                            f"tok={log['mean_tokens']:.0f} | "
                            f"rev={log['mean_reversals']:.1f} | "
                            f"loss={log['loss']:.4f} | "
                            f"{log['elapsed_s']:.1f}s | "
                            f"RAM={mem['available_ram_gb']:.1f}GB"
                        )
                except MemoryGuardError:
                    print("[grpo] Memory guard triggered. Saving and stopping.")
                    self.save_checkpoint()
                    break

                if (step + 1) % save_every == 0:
                    self.save_checkpoint()

        except KeyboardInterrupt:
            print("\n[grpo] Interrupted by user. Saving checkpoint...")
            self.save_checkpoint()

        # Final save
        self.save_checkpoint()

        # Print summary
        if self.log_history:
            print(f"\n{'='*60}")
            print(f"Training complete. {len(self.log_history)} steps logged.")
            recent = self.log_history[-20:]
            print(f"Recent mean outcome: {np.mean([l['mean_outcome'] for l in recent]):.3f}")
            print(f"Recent mean tokens: {np.mean([l['mean_tokens'] for l in recent]):.1f}")
            print(f"Recent mean reversals: {np.mean([l['mean_reversals'] for l in recent]):.2f}")
            print(f"{'='*60}")

        return self.log_history
