"""Model utilities: loading, generation, and logprob computation for GRPO.

Memory-safe: sequential generation, cache cleared after each rollout.
Uses mlx_lm's stream_generate for efficient generation with token capture.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.lora import linear_to_lora_layers
from mlx_lm.sample_utils import make_sampler
from typing import Optional
import json

from .memory_guard import check, MemoryGuardError


def _setup_mlx_memory():
    """Set MLX wired limit to the recommended max for this device."""
    try:
        if mx.metal.is_available():
            max_mem = mx.device_info()["max_recommended_working_set_size"]
            mx.set_wired_limit(max_mem)
            print(f"[model] MLX wired limit set to {max_mem / 1e9:.1f}GB")
    except Exception as e:
        print(f"[model] Could not set wired limit: {e}")


class ModelWrapper:
    """Wraps a model + tokenizer with LoRA, generation, and logprob computation."""

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        lora_config: Optional[dict] = None,
    ):
        print(f"[model] Loading {model_path}...")
        _setup_mlx_memory()
        self.model, self.tokenizer = load(model_path, adapter_path=adapter_path)
        self.model_path = model_path
        self.lora_config = lora_config
        self._lora_applied = False

        if lora_config and not adapter_path:
            self._apply_lora(lora_config)

        self.model.eval()
        mem = check()
        print(f"[model] Loaded. RAM avail: {mem['available_ram_gb']}GB, "
              f"MLX active: {mem['mlx_active_gb']}GB")

    def _apply_lora(self, config: dict):
        """Apply LoRA layers to the model."""
        defaults = {"rank": 8, "alpha": 16, "dropout": 0.0, "scale": 10.0}
        defaults.update(config)
        num_layers = len(self.model.layers) if hasattr(self.model, "layers") else 6
        # Freeze base model first, then add LoRA layers (which are trainable)
        self.model.freeze()
        linear_to_lora_layers(self.model, num_layers, defaults)
        self._lora_applied = True
        from mlx.utils import tree_flatten
        trainable = sum(p.size for _, p in tree_flatten(self.model.trainable_parameters()))
        total = sum(p.size for _, p in tree_flatten(self.model.parameters()))
        print(f"[model] LoRA applied. Trainable: {trainable:,} / {total:,} params")

    def chat_format(self, prompt: str) -> str:
        """Format a prompt using the tokenizer's chat template."""
        messages = [{"role": "user", "content": prompt}]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"User: {prompt}\nAssistant:"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
    ) -> dict:
        """Generate a single completion. Returns text + token ids.

        Sequential, memory-safe: clears cache after generation.
        """
        check()

        formatted = self.chat_format(prompt)
        prompt_ids = self.tokenizer.encode(formatted)

        self.model.eval()
        generated_ids = []
        text_parts = []

        sampler = make_sampler(temp=temperature, top_p=top_p)

        for resp in stream_generate(
            self.model,
            self.tokenizer,
            formatted,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            generated_ids.append(resp.token)
            text_parts.append(resp.text)
            if resp.finish_reason is not None:
                break

        text = "".join(text_parts)
        mx.clear_cache()

        return {
            "text": text,
            "generated_ids": generated_ids,
            "prompt_ids": prompt_ids,
            "token_count": len(generated_ids),
        }

    def compute_logprobs(
        self,
        prompt_ids: list,
        generated_ids: list,
    ) -> mx.array:
        """Compute log-probabilities of generated tokens (with gradients).

        Returns sum of log-probs over generated tokens — used for GRPO:
        loss = -mean(logprob * advantage)
        """
        # Full sequence
        full_ids = prompt_ids + generated_ids
        input_ids = mx.array([full_ids])

        # Forward pass with gradients
        logits = self.model(input_ids)
        logits = logits[0]  # [seq_len, vocab]

        # Position i predicts token i+1
        # Generated token j is at full_ids[len(prompt) + j]
        # Predicted by position len(prompt) + j - 1
        gen_start = len(prompt_ids)
        # Logits at positions [gen_start-1 : gen_start-1+len(gen)]
        pred_logits = logits[gen_start - 1 : gen_start - 1 + len(generated_ids)]
        # Target tokens
        targets = mx.array(generated_ids)

        # Log softmax and gather
        log_probs = nn.log_softmax(pred_logits, axis=-1)
        token_log_probs = log_probs[mx.arange(len(generated_ids)), targets]

        return token_log_probs  # [len(generated_ids)]

    def trainable_params(self):
        """Get trainable (LoRA) parameters as a flat dict."""
        return self.model.trainable_parameters()

    def save_adapters(self, path: str):
        """Save LoRA adapters (trainable params only)."""
        from mlx.utils import tree_flatten
        adapter_weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(path, adapter_weights)
        # Save config alongside
        if self.lora_config:
            cfg_path = path.replace(".safetensors", "_config.json")
            with open(cfg_path, "w") as f:
                json.dump(self.lora_config, f)
        print(f"[model] Saved adapters to {path}")

    def eval_mode(self):
        self.model.eval()

    def train_mode(self):
        self.model.train()
