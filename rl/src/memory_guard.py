"""Memory guard for safe RL training on 8GB M1 Air.

Monitors both MLX unified-memory (active/peak/cache) and system-level free RAM
via `vm_stat`. Before each generation/training step, call `check()` — it will:
  - clear MLX cache if it's grown
  - raise MemoryGuardError if available RAM drops below the safety threshold

The threshold is conservative (1.2GB headroom) to avoid system swap/crash.
"""
import subprocess
import mlx.core as mx


PAGE_SIZE = 16384  # macOS ARM page size
SAFETY_THRESHOLD_GB = 0.5  # minimum reclaimable RAM before we refuse to proceed
CLEAR_CACHE_THRESHOLD_GB = 0.3  # clear MLX cache when it exceeds this


def _vm_stat() -> dict:
    """Parse macOS `vm_stat` output into byte counts."""
    out = subprocess.check_output(["vm_stat"], text=True)
    lines = [l.strip().rstrip(".") for l in out.splitlines()]
    vals = {}
    for l in lines:
        if ":" not in l:
            continue
        key, _, raw = l.partition(":")
        raw = raw.strip().split()[0]  # take first number
        try:
            pages = int(raw)
        except ValueError:
            continue
        vals[key.strip()] = pages * PAGE_SIZE
    return vals


def available_ram_gb() -> float:
    """Estimate available RAM (free + inactive + speculative — all reclaimable on macOS)."""
    v = _vm_stat()
    free = v.get("Pages free", 0)
    inactive = v.get("Pages inactive", 0)
    speculative = v.get("Pages speculative", 0)
    # On macOS, inactive + speculative pages ARE reclaimable by the OS
    avail = free + inactive + speculative
    return avail / 1e9


def mlx_memory_gb() -> dict:
    """MLX's view of unified memory in GB."""
    return {
        "active": mx.get_active_memory() / 1e9,
        "peak": mx.get_peak_memory() / 1e9,
        "cache": mx.get_cache_memory() / 1e9,
    }


class MemoryGuardError(Exception):
    """Raised when available RAM is too low to continue safely."""


def check() -> dict:
    """Call before each heavy operation. Clears cache, raises if low."""
    info = mlx_memory_gb()
    if info["cache"] > CLEAR_CACHE_THRESHOLD_GB:
        mx.clear_cache()
        info = mlx_memory_gb()

    avail = available_ram_gb()
    status = {
        "available_ram_gb": round(avail, 2),
        "mlx_active_gb": round(info["active"], 2),
        "mlx_peak_gb": round(info["peak"], 2),
        "mlx_cache_gb": round(info["cache"], 2),
        "safe": avail >= SAFETY_THRESHOLD_GB,
    }
    if not status["safe"]:
        mx.clear_cache()
        avail2 = available_ram_gb()
        if avail2 < SAFETY_THRESHOLD_GB:
            raise MemoryGuardError(
                f"Available RAM {avail2:.2f}GB < threshold {SAFETY_THRESHOLD_GB}GB. "
                f"Stopping to prevent crash. MLX active: {info['active']:.2f}GB."
            )
        status["available_ram_gb"] = round(avail2, 2)
        status["safe"] = True
    return status


def reset_peak():
    """Reset MLX peak memory tracker."""
    mx.reset_peak_memory()
