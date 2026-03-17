#!/usr/bin/env python3
"""Generate/check/benchmark custom sumcheck cases for student experimentation.

This script is intentionally independent from the public pytest harness.
It lets students create their own vectors (including custom variables) and
validate/benchmark `student.sumcheck` without modifying official tests.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from rich.console import Console
from rich.table import Table

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import student

UINT32_MAX = (1 << 32) - 1
UINT64_MAX = (1 << 64) - 1

PRIMES_32 = [3603169181, 2745059753, 3186843487, 3855252389, 2597694767]
PRIMES_64 = [
    13419220890677911291,
    14450748579811584529,
    16799477333554465393,
    15717685510329205313,
    18297785792242843937,
]


def _mod_add(a, b, q):
    return (int(a) + int(b)) % int(q)


def _mod_sub(a, b, q):
    return (int(a) - int(b)) % int(q)


def _mod_mul(a, b, q):
    return (int(a) * int(b)) % int(q)


def _mod_inv(a, q):
    return pow(int(a) % int(q), int(q) - 2, int(q))


def _lagrange_eval_at(r, y_vals, q):
    degree = len(y_vals) - 1
    r = int(r) % int(q)
    q = int(q)
    acc = 0

    for j in range(degree + 1):
        num = 1
        den = 1
        for m in range(degree + 1):
            if m == j:
                continue
            num = _mod_mul(num, _mod_sub(r, m, q), q)
            den = _mod_mul(den, _mod_sub(j, m, q), q)
        lj = _mod_mul(num, _mod_inv(den, q), q)
        acc = _mod_add(acc, _mod_mul(y_vals[j], lj, q), q)
    return acc


def _verifier_check_and_update_claim(round_evals, claim_prev, challenge_r, q):
    if len(round_evals) < 2:
        raise ValueError("Need at least p(0), p(1)")

    lhs = _mod_add(round_evals[0], round_evals[1], q)
    if lhs != (int(claim_prev) % int(q)):
        return False, None

    claim_next = _lagrange_eval_at(challenge_r, round_evals, q)
    return True, claim_next


def _verifier_sumcheck(round_evals, claim0, challenges, q):
    if len(round_evals) != len(challenges):
        raise ValueError("Need exactly one challenge per round")

    claim = int(claim0) % int(q)
    for p_evals, challenge in zip(round_evals, challenges):
        ok, claim = _verifier_check_and_update_claim(p_evals, claim, challenge, q)
        if not ok:
            return False, None
    return True, claim


def _expr_id(expression):
    return " + ".join("*".join(term) for term in expression)


def _sanitize_case_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", text.strip())
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "custom_case"


def _parse_expression(expr_text: str):
    expr_text = str(expr_text).strip()
    if not expr_text:
        raise ValueError("Expression cannot be empty")

    terms = []
    for raw_term in expr_text.split("+"):
        term_text = raw_term.strip()
        if not term_text:
            raise ValueError(f"Invalid expression term in {expr_text!r}")
        vars_in_term = []
        for raw_var in term_text.split("*"):
            var = raw_var.strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", var):
                raise ValueError(
                    f"Invalid variable name {var!r} in expression {expr_text!r}"
                )
            vars_in_term.append(var)
        terms.append(vars_in_term)

    if not terms:
        raise ValueError(f"Failed to parse expression {expr_text!r}")
    return terms


def _unique_vars_in_expression(expression):
    out = []
    seen = set()
    for term in expression:
        for var in term:
            if var not in seen:
                out.append(var)
                seen.add(var)
    return out


def _parse_var_list(var_list_text: str | None):
    if var_list_text is None:
        return None
    out = []
    seen = set()
    for raw in var_list_text.split(","):
        name = raw.strip()
        if not name:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise ValueError(f"Invalid variable name in --vars: {name!r}")
        if name not in seen:
            out.append(name)
            seen.add(name)
    if not out:
        raise ValueError("--vars was provided but no valid variable names were found")
    return out


def _choose_prime(bits: int, q: int | None, seed: int):
    if q is not None:
        q = int(q)
        if q <= 2:
            raise ValueError("q must be > 2")
        return q

    if int(bits) == 32:
        return int(PRIMES_32[int(seed) % len(PRIMES_32)])
    if int(bits) == 64:
        return int(PRIMES_64[int(seed) % len(PRIMES_64)])
    raise ValueError("bits must be 32 or 64 when --q is not provided")


def _generate_tables(*, variable_names, n: int, q: int, seed: int):
    rng = random.Random(int(seed))
    tables = {}
    for name in variable_names:
        tables[name] = [rng.randrange(int(q)) for _ in range(int(n))]
    return tables


def _generate_challenges(*, rounds: int, q: int, seed: int):
    rng = random.Random(int(seed) + 1_000_003)
    return [rng.randrange(int(q)) for _ in range(int(rounds))]


def _write_tables_npz(path: Path, tables, q: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    if int(q) <= UINT64_MAX:
        payload = {
            name: np.asarray(vals, dtype=np.uint64)
            for name, vals in tables.items()
        }
    else:
        payload = {
            name: np.asarray(vals, dtype=object)
            for name, vals in tables.items()
        }
    np.savez_compressed(path, **payload)


def _read_case(meta_path: Path):
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    required = (
        "id",
        "num_vars",
        "q",
        "expression",
        "variable_names",
        "challenges",
        "table_file",
    )
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"Case metadata missing keys {missing}: {meta_path}")
    return payload


def _load_tables_from_case(case, *, meta_path: Path):
    table_file = case["table_file"]
    table_path = Path(table_file)
    if not table_path.is_absolute():
        table_path = (meta_path.parent / table_file).resolve()
    if not table_path.exists():
        raise FileNotFoundError(f"Table file not found: {table_path}")

    out = {}
    with np.load(table_path, allow_pickle=True) as data:
        for name in case["variable_names"]:
            if name not in data:
                raise ValueError(f"Table file missing variable {name!r}: {table_path}")
            out[name] = [int(v) % int(case["q"]) for v in data[name].tolist()]
    return out


def _eval_at_point(zero_eval, one_eval, target_eval, q):
    q = int(q)
    z = int(zero_eval) % q
    o = int(one_eval) % q
    t = int(target_eval) % q
    return (z + ((o - z) % q) * t) % q


def _eval_table_at_point(table_values, challenges, q):
    cur = [int(v) % int(q) for v in table_values]
    for challenge in challenges:
        if len(cur) % 2 != 0:
            raise ValueError(
                f"Table length must remain even during oracle evaluation; got {len(cur)}"
            )
        next_cur = []
        for i in range(0, len(cur), 2):
            next_cur.append(_eval_at_point(cur[i], cur[i + 1], challenge, q))
        cur = next_cur

    if len(cur) != 1:
        raise ValueError(f"Oracle table reduction expected length=1, got {len(cur)}")
    return int(cur[0]) % int(q)


def _oracle_eval_from_tables(*, expression, tables, challenges, q):
    q = int(q)
    point_values = {
        name: _eval_table_at_point(values, challenges, q)
        for name, values in tables.items()
    }

    acc = 0
    for term in expression:
        term_val = 1
        for var in term:
            if var not in point_values:
                raise KeyError(
                    f"Expression variable {var!r} missing from generated tables"
                )
            term_val = _mod_mul(term_val, point_values[var], q)
        acc = _mod_add(acc, term_val, q)
    return int(acc) % q


def _jax_dtype_for_q(q: int):
    if int(q) <= UINT32_MAX:
        return jnp.uint32
    if int(q) <= UINT64_MAX:
        return jnp.uint64
    raise ValueError("JAX mode supports only q <= 2^64-1")


def _normalize_round_evals(raw, q):
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    if len(raw) > 0 and isinstance(raw[0], list):
        return [[int(v) % q for v in row] for row in raw]
    return [int(v) % q for v in raw]


def cmd_generate(args):
    expression = _parse_expression(args.expression)
    expr_vars = _unique_vars_in_expression(expression)
    provided_vars = _parse_var_list(args.vars)
    variable_names = provided_vars if provided_vars is not None else expr_vars

    missing_vars = [v for v in expr_vars if v not in variable_names]
    if missing_vars:
        raise ValueError(
            f"Variables from expression missing in --vars: {missing_vars}"
        )

    q = _choose_prime(args.bits, args.q, args.seed)
    num_vars = int(args.num_vars)
    if num_vars <= 0:
        raise ValueError("--num-vars must be > 0")
    n = 1 << num_vars

    case_id = args.case_id
    if case_id is None:
        expr_tag = _sanitize_case_id(_expr_id(expression))
        case_id = f"custom_v{num_vars}_{args.bits}b_{expr_tag}_s{int(args.seed)}"

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / f"{case_id}.npz"
    meta_path = out_dir / f"{case_id}_meta.json"

    if (table_path.exists() or meta_path.exists()) and not args.overwrite:
        raise FileExistsError(
            f"Case already exists ({meta_path.name}). Use --overwrite to replace it."
        )

    print(
        f"[custom-generate] case_id={case_id} num_vars={num_vars} "
        f"table_length={n} q={q} vars={variable_names} expr={_expr_id(expression)!r}"
    )
    t0 = time.perf_counter()
    tables = _generate_tables(variable_names=variable_names, n=n, q=q, seed=args.seed)
    challenges = _generate_challenges(rounds=num_vars, q=q, seed=args.seed)
    _write_tables_npz(table_path, tables, q=q)

    payload = {
        "id": str(case_id),
        "num_vars": int(num_vars),
        "prime_bits": int(args.bits),
        "q": int(q),
        "variable_names": [str(v) for v in variable_names],
        "expression": expression,
        "challenges": [int(v) % int(q) for v in challenges],
        "table_file": table_path.name,
        "generated_by": "scripts/custom_cases.py",
        "seed": int(args.seed),
    }
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    print(f"[custom-generate] wrote {meta_path} in {elapsed:.2f}s")
    print(
        "[custom-generate]\n  run correctness checks with:\n"
        f"  uv run python scripts/custom_cases.py check --case-meta {meta_path}\n"
        f"  or\n  uv run python scripts/custom_cases.py check --case-id {case_id}\n"
        "\n"
        "  benchmark with:\n"
        f"  uv run python scripts/custom_cases.py bench --case-id {case_id} --runs 8 --warmup 3"
    )


def _resolve_meta_path(case_meta: str | None, case_id: str | None, out_dir: str):
    if case_meta:
        return Path(case_meta).resolve()
    if case_id:
        return (Path(out_dir).resolve() / f"{case_id}_meta.json").resolve()
    raise ValueError("Provide either --case-meta or --case-id")


def _run_student_once(case, *, meta_path: Path):
    q = int(case["q"])
    bit_width = int(case.get("prime_bits", 32))
    dtype = _jax_dtype_for_q(q)
    q_arg = jnp.asarray(q, dtype=dtype)

    tables = _load_tables_from_case(case, meta_path=meta_path)
    eval_tables = {
        name: jnp.asarray(vals, dtype=dtype)
        for name, vals in tables.items()
    }
    challenges_full = [int(v) % q for v in case["challenges"]]
    num_rounds = len(challenges_full)
    prover_challenges = jnp.asarray(challenges_full[:-1], dtype=dtype)

    out = student.sumcheck(
        eval_tables,
        q=q_arg,
        expression=case["expression"],
        challenges=prover_challenges,
        num_rounds=num_rounds,
        bit_width=bit_width,
    )
    jax.block_until_ready(out)

    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise TypeError(
            "student.sumcheck must return (claim0, round_evals) for custom check"
        )
    claim0_raw, round_evals_raw = out
    claim0 = int(claim0_raw) % q
    round_evals = _normalize_round_evals(round_evals_raw, q)

    return {
        "claim0": claim0,
        "round_evals": round_evals,
        "challenges_full": challenges_full,
        "q": q,
    }


def cmd_check(args):
    meta_path = _resolve_meta_path(args.case_meta, args.case_id, args.out_dir)
    case = _read_case(meta_path)
    q = int(case["q"])
    tables = _load_tables_from_case(case, meta_path=meta_path)

    t0 = time.perf_counter()
    got = _run_student_once(case, meta_path=meta_path)
    elapsed = time.perf_counter() - t0

    expected_rounds = len(got["challenges_full"])
    if len(got["round_evals"]) != expected_rounds:
        raise AssertionError(
            f"round count mismatch: got {len(got['round_evals'])}, "
            f"expected {expected_rounds}"
        )

    ok, final_claim = _verifier_sumcheck(
        got["round_evals"],
        got["claim0"],
        got["challenges_full"],
        q,
    )
    if not ok:
        raise AssertionError("Verifier consistency check failed on student transcript")

    oracle_eval = _oracle_eval_from_tables(
        expression=case["expression"],
        tables=tables,
        challenges=got["challenges_full"],
        q=q,
    )
    if int(final_claim) % q != int(oracle_eval) % q:
        raise AssertionError(
            "Final oracle check failed: "
            f"final_claim={int(final_claim) % q}, oracle_eval={int(oracle_eval) % q}"
        )

    print(
        f"[custom-check] PASS case={case['id']} expr={_expr_id(case['expression'])!r} "
        f"final_claim={int(final_claim) % q} oracle_eval={int(oracle_eval) % q} "
        f"elapsed={elapsed:.3f}s"
    )


def cmd_bench(args):
    meta_path = _resolve_meta_path(args.case_meta, args.case_id, args.out_dir)
    case = _read_case(meta_path)
    q = int(case["q"])
    bit_width = int(case.get("prime_bits", 32))
    dtype = _jax_dtype_for_q(q)
    q_arg = jnp.asarray(q, dtype=dtype)

    tables = _load_tables_from_case(case, meta_path=meta_path)
    n = len(next(iter(tables.values())))
    eval_tables = {
        name: jnp.asarray(vals, dtype=dtype)
        for name, vals in tables.items()
    }
    challenges_full = [int(v) % q for v in case["challenges"]]
    prover_challenges = jnp.asarray(challenges_full[:-1], dtype=dtype)
    num_rounds = len(challenges_full)
    expression = case["expression"]

    fn = jax.jit(
        lambda t, r: student.sumcheck(
            t,
            q=q_arg,
            expression=expression,
            challenges=r,
            num_rounds=num_rounds,
            bit_width=bit_width,
        )
    )

    t0 = time.perf_counter()
    out = fn(eval_tables, prover_challenges)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    warmup_times = []
    for _ in range(max(0, int(args.warmup) - 1)):
        t0 = time.perf_counter()
        out = fn(eval_tables, prover_challenges)
        jax.block_until_ready(out)
        warmup_times.append(time.perf_counter() - t0)

    timed_times = []
    for _ in range(int(args.runs)):
        t0 = time.perf_counter()
        out = fn(eval_tables, prover_challenges)
        jax.block_until_ready(out)
        timed_times.append(time.perf_counter() - t0)

    arr = np.asarray(timed_times, dtype=np.float64)
    median_s = float(np.median(arr))
    p90_s = float(np.quantile(arr, 0.90))
    min_s = float(np.min(arr))
    max_s = float(np.max(arr))
    throughput = (n / median_s / 1e6) if median_s > 0 else float("inf")

    console = Console()
    table = Table(title="Custom Sumcheck Latency")
    table.add_column("case", justify="left")
    table.add_column("bits", justify="right")
    table.add_column("N", justify="right")
    table.add_column("expr", justify="left")
    table.add_column("compile (ms)", justify="right")
    table.add_column("median (ms)", justify="right")
    table.add_column("p90 (ms)", justify="right")
    table.add_column("min (ms)", justify="right")
    table.add_column("max (ms)", justify="right")
    table.add_column("Mpts/s", justify="right")
    table.add_row(
        str(case["id"]),
        str(case.get("prime_bits", "?")),
        str(n),
        _expr_id(expression),
        f"{compile_s * 1e3:.3f}",
        f"{median_s * 1e3:.3f}",
        f"{p90_s * 1e3:.3f}",
        f"{min_s * 1e3:.3f}",
        f"{max_s * 1e3:.3f}",
        f"{throughput:.2f}",
    )
    console.print(table)

    if args.show_invocation_times:
        details = Table(title="Custom Per-Invocation Times")
        details.add_column("case", justify="left")
        details.add_column("expr", justify="left")
        details.add_column("phase", justify="left")
        details.add_column("iter", justify="right")
        details.add_column("elapsed (ms)", justify="right")
        details.add_row(str(case["id"]), _expr_id(expression), "compile", "0", f"{compile_s * 1e3:.3f}")
        for idx, seconds in enumerate(warmup_times, start=1):
            details.add_row(
                str(case["id"]),
                _expr_id(expression),
                "warmup",
                str(idx),
                f"{seconds * 1e3:.3f}",
            )
        for idx, seconds in enumerate(timed_times, start=1):
            details.add_row(
                str(case["id"]),
                _expr_id(expression),
                "timed",
                str(idx),
                f"{seconds * 1e3:.3f}",
            )
        console.print(details)


def build_parser():
    p = argparse.ArgumentParser(
        description="Generate/check/benchmark custom sumcheck cases."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate", help="Generate one custom case")
    p_gen.add_argument("--expression", type=str, required=True, help='e.g. "a*b + c"')
    p_gen.add_argument("--num-vars", type=int, required=True, help="Number of rounds/vars")
    p_gen.add_argument("--bits", type=int, default=32, choices=[32, 64])
    p_gen.add_argument("--q", type=int, default=None, help="Explicit prime modulus")
    p_gen.add_argument("--seed", type=int, default=0)
    p_gen.add_argument("--vars", type=str, default=None, help="Comma list, e.g. a,b,c,d,e,g,h")
    p_gen.add_argument("--case-id", type=str, default=None)
    p_gen.add_argument("--out-dir", type=str, default="tests/data/custom")
    p_gen.add_argument("--overwrite", action="store_true")
    p_gen.set_defaults(fn=cmd_generate)

    p_chk = sub.add_parser("check", help="Check student.sumcheck on one custom case")
    p_chk.add_argument("--case-meta", type=str, default=None)
    p_chk.add_argument("--case-id", type=str, default=None)
    p_chk.add_argument("--out-dir", type=str, default="tests/data/custom")
    p_chk.set_defaults(fn=cmd_check)

    p_bench = sub.add_parser("bench", help="Benchmark student.sumcheck on one custom case")
    p_bench.add_argument("--case-meta", type=str, default=None)
    p_bench.add_argument("--case-id", type=str, default=None)
    p_bench.add_argument("--out-dir", type=str, default="tests/data/custom")
    p_bench.add_argument("--runs", type=int, default=8)
    p_bench.add_argument("--warmup", type=int, default=3)
    p_bench.add_argument("--show-invocation-times", action="store_true")
    p_bench.set_defaults(fn=cmd_bench)

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
