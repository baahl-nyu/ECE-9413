"""Benchmark runner for Assignment 2 sumcheck (JAX-focused)."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.pop("LD_LIBRARY_PATH", None)

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

from sumcheck_utils import EXPRESSION_IDS, expression_from_id, expression_to_id
import student
from tests.case_selection import select_cases
from tests.case_utils import iter_case_expressions, load_tables
from tests.data_loader import discover_cases

DEFAULT_RUNS = 8
DEFAULT_WARMUP = 3


@dataclass(frozen=True)
class BenchStats:
    compile_s: float
    median_s: float
    p90_s: float
    min_s: float
    max_s: float


@dataclass(frozen=True)
class BenchTrace:
    compile_s: float
    warmup_s: list[float]
    timed_s: list[float]


def summarize(compile_s, times):
    arr = np.asarray(times, dtype=np.float64)
    return BenchStats(
        compile_s=float(compile_s),
        median_s=float(np.median(arr)),
        p90_s=float(np.quantile(arr, 0.90)),
        min_s=float(np.min(arr)),
        max_s=float(np.max(arr)),
    )


def _to_jax_tables(tables, q):
    if int(q) <= (1 << 32) - 1:
        dtype = jnp.uint32
    elif int(q) <= (1 << 64) - 1:
        dtype = jnp.uint64
    else:
        raise ValueError(
            "JAX benchmark supports only q <= 2^64-1. "
            "Use 32/64-bit cases for benchmark mode."
        )

    out = {}
    for name, values in tables.items():
        out[name] = jnp.asarray(values, dtype=dtype)
    return out


def _to_jax_challenges(challenges, q):
    if int(q) <= (1 << 32) - 1:
        return jnp.asarray(challenges, dtype=jnp.uint32)
    return jnp.asarray(challenges, dtype=jnp.uint64)


def _collect_requested_bits(bits=None, all_32=False, all_64=False):
    requested_bits = [str(v) for v in (bits or [])]
    if all_32:
        requested_bits.append("32")
    if all_64:
        requested_bits.append("64")
    return requested_bits


def _explicit_64_selected(case_ids=None, requested_bits=None):
    if any(str(v) == "64" for v in (requested_bits or [])):
        return True
    return any("_case64_" in str(case_id) for case_id in (case_ids or []))


def choose_cases(
    *,
    case_ids=None,
    bits=None,
    num_vars=None,
    all_32=False,
    all_64=False,
    enable_challenge32=False,
    enable_core64=False,
    enable_challenge64=False,
):
    cases = discover_cases()
    if not cases:
        raise RuntimeError(
            "No test cases found in tests/data. "
            "Ensure your assignment release includes tests/data case files."
        )

    requested_ids = list(case_ids or [])
    if requested_ids:
        known_ids = {str(case.get("id", "")) for case in cases}
        missing_ids = [cid for cid in requested_ids if cid not in known_ids]
        if missing_ids:
            raise ValueError(f"Unknown case id(s): {missing_ids}")

    requested_bits = _collect_requested_bits(bits=bits, all_32=all_32, all_64=all_64)
    explicit_64_requested = _explicit_64_selected(
        case_ids=requested_ids,
        requested_bits=requested_bits,
    )

    selected = select_cases(
        cases,
        case_ids=requested_ids,
        bits=requested_bits,
        num_vars=list(num_vars or []),
        enable_challenge32=bool(enable_challenge32),
        enable_core64=bool(enable_core64 or explicit_64_requested),
        enable_challenge64=bool(enable_challenge64),
    )

    if not selected:
        raise ValueError(
            "No benchmark cases matched filters. "
            "Check --case-id/--bits/--num-vars/track flags."
        )

    explicit_case_filters = bool(requested_ids or requested_bits or (num_vars or []))
    if explicit_case_filters:
        return selected

    # Default benchmark mode: one smallest compulsory case for quick feedback.
    selected = sorted(
        selected,
        key=lambda c: (
            int(c.get("num_vars", 10**9)),
            int(c.get("prime_bits", 10**9)),
            str(c.get("id", "")),
        ),
    )
    return [selected[0]]


def _block_output(out):
    try:
        jax.block_until_ready(out)
    except Exception:
        pass


def bench_single(case, expression, runs, warmup):
    q = int(case["q"])
    bit_width = int(case.get("prime_bits", 32))
    all_challenges = [int(x) % q for x in case["challenges"]]
    num_rounds = len(all_challenges)
    challenges = all_challenges[:-1]
    tests_dir = Path(__file__).resolve().parent
    eval_tables = load_tables(case, tests_dir=tests_dir)

    eval_tables = _to_jax_tables(eval_tables, q)
    challenges = _to_jax_challenges(challenges, q)

    fn = jax.jit(
        lambda tables, rs: student.sumcheck(
            tables,
            q=q,
            expression=expression,
            challenges=rs,
            num_rounds=num_rounds,
            bit_width=bit_width,
        )
    )

    t0 = time.perf_counter()
    out = fn(eval_tables, challenges)
    _block_output(out)
    compile_s = time.perf_counter() - t0

    warmup_times = []
    for _ in range(max(0, warmup - 1)):
        t0 = time.perf_counter()
        out = fn(eval_tables, challenges)
        _block_output(out)
        warmup_times.append(time.perf_counter() - t0)

    timed_times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = fn(eval_tables, challenges)
        _block_output(out)
        timed_times.append(time.perf_counter() - t0)

    return summarize(compile_s, timed_times), BenchTrace(
        compile_s=compile_s,
        warmup_s=warmup_times,
        timed_s=timed_times,
    )


def run_correctness(
    case_ids=None,
    bits=None,
    num_vars=None,
    all_32=False,
    all_64=False,
    enable_challenge32=False,
    enable_core64=False,
    enable_challenge64=False,
    include_128=False,
):
    cmd = ["uv", "run", "pytest"]

    for case_id in (case_ids or []):
        cmd.extend(["--case-id", case_id])
    for bit in (bits or []):
        cmd.extend(["--bits", str(bit)])
    for nv in (num_vars or []):
        cmd.extend(["--num-vars", str(nv)])
    if all_32:
        cmd.append("--all-32")
    if all_64:
        cmd.append("--all-64")
    if enable_challenge32:
        cmd.append("--enable-challenge32")
    if enable_core64:
        cmd.append("--enable-core64")
    if enable_challenge64:
        cmd.append("--enable-challenge64")
    if include_128:
        cmd.append("--include-128")

    subprocess.run(cmd, check=True)


def run_latency(
    *,
    case_ids=None,
    bits=None,
    num_vars=None,
    all_32=False,
    all_64=False,
    enable_challenge32=False,
    enable_core64=False,
    enable_challenge64=False,
    expression_id=None,
    runs=DEFAULT_RUNS,
    warmup=DEFAULT_WARMUP,
    show_invocation_times=False,
):
    console = Console()

    cases = choose_cases(
        case_ids=case_ids,
        bits=bits,
        num_vars=num_vars,
        all_32=all_32,
        all_64=all_64,
        enable_challenge32=enable_challenge32,
        enable_core64=enable_core64,
        enable_challenge64=enable_challenge64,
    )

    device = jax.devices()[0]
    console.print(f"Device: {device.platform} ({device.device_kind})")

    table = Table(title="Sumcheck Latency")
    table.add_column("testcase", justify="left")
    table.add_column("bits", justify="right")
    table.add_column("N", justify="right")
    table.add_column("expr", justify="left")
    table.add_column("compile (ms)", justify="right")
    table.add_column("median (ms)", justify="right")
    table.add_column("p90 (ms)", justify="right")
    table.add_column("Mpts/s", justify="right")

    invocation_rows = []

    for case in cases:
        supported = iter_case_expressions(case)
        supported_ids = [expression_to_id(e) for e in supported]

        if expression_id is None:
            expression_ids = supported_ids
        else:
            if expression_id not in supported_ids:
                raise ValueError(
                    f"Expression {expression_id!r} not available in case "
                    f"{case['id']}. Available: {supported_ids}"
                )
            expression_ids = [expression_id]

        q = int(case["q"])
        n = len(load_tables(case, tests_dir=Path(__file__).resolve().parent)["a"])

        for expr_id in expression_ids:
            expression = expression_from_id(expr_id)
            stats, trace = bench_single(
                case=case,
                expression=expression,
                runs=runs,
                warmup=warmup,
            )
            throughput = n / stats.median_s / 1e6

            table.add_row(
                case["id"],
                str(case.get("prime_bits", "?")),
                str(n),
                expr_id,
                f"{stats.compile_s * 1e3:.2f}",
                f"{stats.median_s * 1e3:.3f}",
                f"{stats.p90_s * 1e3:.3f}",
                f"{throughput:.2f}",
            )

            if show_invocation_times:
                invocation_rows.append(
                    (case["id"], expr_id, "compile", 0, trace.compile_s * 1e3)
                )
                for idx, seconds in enumerate(trace.warmup_s, start=1):
                    invocation_rows.append(
                        (case["id"], expr_id, "warmup", idx, seconds * 1e3)
                    )
                for idx, seconds in enumerate(trace.timed_s, start=1):
                    invocation_rows.append(
                        (case["id"], expr_id, "timed", idx, seconds * 1e3)
                    )

    console.print(table)

    if show_invocation_times and invocation_rows:
        details = Table(title="Per-Invocation Times")
        details.add_column("testcase", justify="left")
        details.add_column("expr", justify="left")
        details.add_column("phase", justify="left")
        details.add_column("iter", justify="right")
        details.add_column("elapsed (ms)", justify="right")

        for case_id, expr_id, phase, idx, elapsed_ms in invocation_rows:
            details.add_row(
                str(case_id),
                str(expr_id),
                str(phase),
                str(idx),
                f"{elapsed_ms:.3f}",
            )

        console.print(details)


def build_parser():
    p = argparse.ArgumentParser(
        description="Run correctness tests and/or sumcheck latency benchmark."
    )
    p.add_argument("--tests", action="store_true", help="Run pytest")
    p.add_argument("--bench", action="store_true", help="Run latency benchmark")
    p.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Case id(s) to run (repeatable)",
    )
    p.add_argument(
        "--bits",
        action="append",
        default=[],
        help="Case bit filter for tests/bench (32/64, repeatable)",
    )
    p.add_argument(
        "--num-vars",
        action="append",
        default=[],
        help="Variable-count filter for tests/bench (repeatable)",
    )
    p.add_argument(
        "--all-32",
        action="store_true",
        help="Select all 32-bit cases in tests/bench mode.",
    )
    p.add_argument(
        "--all-64",
        action="store_true",
        help="Select all 64-bit cases in tests/bench mode.",
    )
    p.add_argument(
        "--enable-challenge32",
        action="store_true",
        help="Include challenge expressions on 32-bit cases.",
    )
    p.add_argument(
        "--enable-core64",
        action="store_true",
        help="Include core expressions on 64-bit cases.",
    )
    p.add_argument(
        "--enable-challenge64",
        action="store_true",
        help="Include challenge expressions on 64-bit cases.",
    )
    p.add_argument(
        "--include-128",
        action="store_true",
        help="Include optional 128-bit cases in correctness mode",
    )
    p.add_argument(
        "--expr",
        type=str,
        default=None,
        choices=EXPRESSION_IDS,
        help="Optional single expression id to benchmark (default: all for case)",
    )
    p.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    p.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    p.add_argument(
        "--show-invocation-times",
        action="store_true",
        help="Print timing for every compile/warmup/timed invocation per case/expression.",
    )
    return p


def main():
    args = build_parser().parse_args()

    do_tests = args.tests
    do_bench = args.bench

    if not do_tests and not do_bench:
        do_tests = True
        do_bench = True

    if do_tests:
        run_correctness(
            case_ids=args.case_id,
            bits=args.bits,
            num_vars=args.num_vars,
            all_32=bool(args.all_32),
            all_64=bool(args.all_64),
            enable_challenge32=bool(args.enable_challenge32),
            enable_core64=bool(args.enable_core64),
            enable_challenge64=bool(args.enable_challenge64),
            include_128=args.include_128,
        )

    if do_bench:
        run_latency(
            case_ids=args.case_id,
            bits=args.bits,
            num_vars=args.num_vars,
            all_32=bool(args.all_32),
            all_64=bool(args.all_64),
            enable_challenge32=bool(args.enable_challenge32),
            enable_core64=bool(args.enable_core64),
            enable_challenge64=bool(args.enable_challenge64),
            expression_id=args.expr,
            runs=args.runs,
            warmup=args.warmup,
            show_invocation_times=bool(args.show_invocation_times),
        )


if __name__ == "__main__":
    main()
