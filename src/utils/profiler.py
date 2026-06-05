"""Pipeline profiler: per-stage timing and backbone FLOPs/parameter counts."""

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class _StageStats:
    total_ms: float = 0.0
    count: int = 0

    @property
    def avg_ms(self) -> float:
        return self.total_ms / self.count if self.count > 0 else 0.0


class PipelineProfiler:
    """Accumulates per-stage timing and backbone FLOPs / parameter counts.

    Timing uses torch.cuda.Event on CUDA (gives device-accurate wall time
    with synchronization) and time.perf_counter on CPU.
    FLOPs are measured with thop for PyTorch models and onnx for ONNX models.
    """

    def __init__(self, device: str = "cpu"):
        self._use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self._stats: dict[str, _StageStats] = defaultdict(_StageStats)
        self._model_profiles: dict[str, dict] = {}

    # ── Timing ────────────────────────────────────────────────────────

    @contextmanager
    def time_stage(self, name: str):
        """Context manager that records wall-time for a named pipeline stage."""
        if self._use_cuda:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            try:
                yield
            finally:
                end_event.record()
                torch.cuda.synchronize()
                elapsed_ms = start_event.elapsed_time(end_event)
        else:
            t0 = time.perf_counter()
            try:
                yield
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

        stats = self._stats[name]
        stats.total_ms += elapsed_ms
        stats.count += 1

    # ── FLOPs / parameter profiling ───────────────────────────────────

    def profile_torch_model(
        self,
        name: str,
        model: torch.nn.Module,
        input_tensor: torch.Tensor,
    ) -> None:
        """Profile a PyTorch model with thop (MACs reported as FLOPs × 2)."""
        try:
            from thop import profile as thop_profile  # type: ignore

            model.eval()
            flops, params = thop_profile(
                model, inputs=(input_tensor,), verbose=False
            )
            # thop returns MACs; multiply by 2 to get FLOPs
            self._model_profiles[name] = {
                "flops": int(flops * 2),
                "params": int(params),
            }
        except Exception as exc:
            self._model_profiles[name] = {
                "flops": None,
                "params": None,
                "error": str(exc),
            }

    def profile_onnx_model(self, name: str, model_path: str) -> None:
        """Count parameters in an ONNX model (FLOPs left as N/A — ONNX graphs
        lack the op-level metadata thop needs)."""
        try:
            import numpy as np
            import onnx  # type: ignore

            proto = onnx.load(model_path)
            params = sum(
                int(np.prod(init.dims))
                for init in proto.graph.initializer
                if len(init.dims) > 0
            )
            self._model_profiles[name] = {"flops": None, "params": params}
        except Exception as exc:
            self._model_profiles[name] = {
                "flops": None,
                "params": None,
                "error": str(exc),
            }

    def record_model_profile(
        self,
        name: str,
        flops: Optional[int],
        params: Optional[int],
    ) -> None:
        """Manually record FLOPs and parameter counts (e.g. from vendor APIs)."""
        self._model_profiles[name] = {"flops": flops, "params": params}

    # ── Reporting ─────────────────────────────────────────────────────

    @staticmethod
    def _fmt(n: Optional[int]) -> str:
        if n is None:
            return "N/A"
        for threshold, suffix in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "K")):
            if n >= threshold:
                return f"{n / threshold:.2f}{suffix}"
        return str(n)

    def print_summary(self) -> None:
        sep = "=" * 68
        print(f"\n{sep}")
        print("  PIPELINE PROFILING SUMMARY")
        print(sep)

        if self._stats:
            hdr = f"{'Stage':<22} {'Calls':>7} {'Total ms':>11} {'Avg ms':>10}"
            print(f"\n{hdr}")
            print("-" * 54)
            grand_total = 0.0
            for stage, s in self._stats.items():
                print(
                    f"{stage:<22} {s.count:>7} {s.total_ms:>11.1f} {s.avg_ms:>10.3f}"
                )
                grand_total += s.total_ms
            print("-" * 54)
            print(f"{'TOTAL':<22} {'':>7} {grand_total:>11.1f}")

        if self._model_profiles:
            print(f"\n{'Backbone':<22} {'FLOPs':>12} {'Params':>12}")
            print("-" * 48)
            for bname, p in self._model_profiles.items():
                err = p.get("error")
                if err:
                    print(f"{bname:<22}  error: {err[:40]}")
                else:
                    print(
                        f"{bname:<22} {self._fmt(p.get('flops')):>12}"
                        f" {self._fmt(p.get('params')):>12}"
                    )

        print(f"{sep}\n")
