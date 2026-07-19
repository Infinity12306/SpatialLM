#!/usr/bin/env python3
"""Controlled GPU memory and utilization stress test."""

from __future__ import annotations

import argparse
import math
import time

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Allocate a requested amount of GPU memory and run repeated matrix "
            "multiplications for a fixed duration."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory_gb", type=float, default=77.32)
    parser.add_argument("--duration", type=float, required=True, help="Run time in seconds.")
    parser.add_argument("--matmul_size", type=int, default=4096)
    parser.add_argument("--sleep", type=float, default=0.0, help="Optional sleep between iterations.")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    args = parser.parse_args()
    if args.memory_gb <= 0:
        parser.error("--memory_gb must be positive.")
    if args.duration <= 0:
        parser.error("--duration must be positive.")
    if args.matmul_size <= 0:
        parser.error("--matmul_size must be positive.")
    if args.sleep < 0:
        parser.error("--sleep must be non-negative.")
    return args


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def gb(value: int) -> float:
    return value / 1024**3


def allocate_memory(device: torch.device, memory_gb: float) -> list[torch.Tensor]:
    target_bytes = int(memory_gb * 1024**3)
    tensors: list[torch.Tensor] = []
    chunk_bytes = 1 * 1024**3
    remaining = target_bytes

    while remaining > 0:
        cur_bytes = min(chunk_bytes, remaining)
        numel = max(1, cur_bytes // torch.empty((), dtype=torch.uint8).element_size())
        tensors.append(torch.empty(numel, dtype=torch.uint8, device=device))
        remaining -= numel

    return tensors


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dtype = dtype_from_name(args.dtype)

    print(
        f"Allocating {args.memory_gb:.2f} GiB on {device} for {args.duration:.1f}s; "
        f"matmul_size={args.matmul_size}, dtype={args.dtype}",
        flush=True,
    )
    holders = allocate_memory(device, args.memory_gb)
    torch.cuda.synchronize(device)
    print(
        f"after allocation: allocated={gb(torch.cuda.memory_allocated(device)):.2f}GiB "
        f"reserved={gb(torch.cuda.memory_reserved(device)):.2f}GiB",
        flush=True,
    )

    a = torch.randn(args.matmul_size, args.matmul_size, device=device, dtype=dtype)
    b = torch.randn(args.matmul_size, args.matmul_size, device=device, dtype=dtype)
    c = torch.empty(args.matmul_size, args.matmul_size, device=device, dtype=dtype)

    end_time = time.monotonic() + args.duration
    iterations = 0
    last_report = time.monotonic()
    while time.monotonic() < end_time:
        c = torch.matmul(a, b)
        a = torch.sin(c)
        iterations += 1
        if args.sleep:
            torch.cuda.synchronize(device)
            time.sleep(args.sleep)
        now = time.monotonic()
        if now - last_report >= 30:
            torch.cuda.synchronize(device)
            remaining = max(0.0, end_time - now)
            print(
                f"iter={iterations} remaining={remaining:.1f}s "
                f"allocated={gb(torch.cuda.memory_allocated(device)):.2f}GiB "
                f"reserved={gb(torch.cuda.memory_reserved(device)):.2f}GiB",
                flush=True,
            )
            last_report = now

    torch.cuda.synchronize(device)
    elapsed = args.duration
    print(
        f"done iterations={iterations} approx_iter_per_sec={iterations / max(elapsed, 1e-6):.3f}",
        flush=True,
    )

    del holders, a, b, c
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
