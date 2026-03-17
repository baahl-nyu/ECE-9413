"""Public tests for Assignment 2 sumcheck implementation."""

from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pytest

from sumcheck_utils import expression_to_id
import student
from tests.case_utils import expected_entry, iter_case_expressions, load_tables

MAX_PRIME_32 = 4294967291
MAX_PRIME_64 = 18446744073709551557
MAX_PRIME_128 = 340282366920938463463374607431768211297


def _to_jax_tables(tables, q):
    if int(q) <= (1 << 32) - 1:
        dtype = jnp.uint32
    elif int(q) <= (1 << 64) - 1:
        dtype = jnp.uint64
    else:
        # Optional 128-bit track may require non-JAX bigint logic.
        return tables

    out = {}
    for name, values in tables.items():
        out[name] = jnp.asarray(values, dtype=dtype)
    return out


def _to_jax_challenges(challenges, q):
    if int(q) <= (1 << 32) - 1:
        return jnp.asarray(challenges, dtype=jnp.uint32)
    if int(q) <= (1 << 64) - 1:
        return jnp.asarray(challenges, dtype=jnp.uint64)
    return challenges


def _to_jax_q(q):
    q = int(q)
    if q <= (1 << 32) - 1:
        return jnp.asarray(q, dtype=jnp.uint32)
    if q <= (1 << 64) - 1:
        return jnp.asarray(q, dtype=jnp.uint64)
    return q


def _normalize_round_evals(raw, q):
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    elif isinstance(raw, tuple):
        raw = list(raw)

    if len(raw) > 0 and isinstance(raw[0], (list, tuple)):
        round_evals = [[int(v) % q for v in row] for row in raw]
    else:
        round_evals = [int(v) % q for v in raw]
    return round_evals


def _extract_claim0_and_round_evals(result, q, *, case_id: str, expr_id: str):
    """Require student output to be `(claim0, round_evals)` as JAX arrays."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise AssertionError(
            f"{case_id}/{expr_id}: sumcheck must return "
            f"(claim0, round_evals) as a 2-tuple/list"
        )

    claim0, round_evals = result
    if not isinstance(claim0, jax.Array):
        raise AssertionError(
            f"{case_id}/{expr_id}: claim0 must be a jnp array "
            f"(got {type(claim0).__name__})"
        )
    if not isinstance(round_evals, jax.Array):
        raise AssertionError(
            f"{case_id}/{expr_id}: round_evals must be a jnp array "
            f"(got {type(round_evals).__name__})"
        )

    return int(claim0) % q, _normalize_round_evals(round_evals, q)


def _lagrange_eval_at(r, y_vals, q):
    degree = len(y_vals) - 1
    r = int(r) % q
    acc = 0

    for j in range(degree + 1):
        num = 1
        den = 1
        for m in range(degree + 1):
            if m == j:
                continue
            num = (num * ((r - m) % q)) % q
            den = (den * ((j - m) % q)) % q
        lj = (num * pow(den, q - 2, q)) % q
        acc = (acc + (int(y_vals[j]) % q) * lj) % q

    return acc


def _verifier_consistency_check(round_evals, claim0, challenges, q):
    if len(round_evals) != len(challenges):
        return False, None

    claim = int(claim0) % q
    for p_evals, challenge in zip(round_evals, challenges):
        if not isinstance(p_evals, list) or len(p_evals) < 2:
            return False, None

        lhs = (int(p_evals[0]) + int(p_evals[1])) % q
        if lhs != claim:
            return False, None

        claim = _lagrange_eval_at(challenge, p_evals, q)

    return True, claim


def _scalar_to_int(value):
    if hasattr(value, "shape") and tuple(value.shape) == ():
        return int(value)
    if hasattr(value, "tolist"):
        out = value.tolist()
        if isinstance(out, list):
            raise AssertionError(f"Expected scalar output, got list: {out}")
        return int(out)
    return int(value)


def _eval_mod_op(fn, a, b, q, *, input_dtype=None):
    q_int = int(q)
    a_int = int(a) % q_int
    b_int = int(b) % q_int
    if input_dtype is None:
        q_arg = q_int
        a_arg = a_int
        b_arg = b_int
    else:
        q_arg = jnp.asarray(q_int, dtype=input_dtype)
        a_arg = jnp.asarray(a_int, dtype=input_dtype)
        b_arg = jnp.asarray(b_int, dtype=input_dtype)
    return _scalar_to_int(fn(a_arg, b_arg, q_arg)) % q_int


EDGE_CASES_32 = [
    pytest.param(0, 0, id="zero_zero"),
    pytest.param(1, 2, id="a_lt_b_small"),
    pytest.param(2, 1, id="a_gt_b_small"),
    pytest.param(MAX_PRIME_32 - 1, 1, id="add_wrap_exact_q"),
    pytest.param(MAX_PRIME_32 - 2, 3, id="add_wrap_over_q"),
    pytest.param(1, MAX_PRIME_32 - 1, id="sub_negative_wrap"),
    pytest.param(MAX_PRIME_32 - 1, MAX_PRIME_32 - 1, id="overflow_add_full"),
    pytest.param(MAX_PRIME_32 - 2, MAX_PRIME_32 - 2, id="overflow_add_near_full"),
    pytest.param(MAX_PRIME_32 // 2, MAX_PRIME_32 // 3, id="mid_values"),
    pytest.param(MAX_PRIME_32 - 12345, MAX_PRIME_32 - 6789, id="high_values"),
]

EDGE_CASES_64 = [
    pytest.param(0, 0, id="zero_zero"),
    pytest.param(1, 2, id="a_lt_b_small"),
    pytest.param(2, 1, id="a_gt_b_small"),
    pytest.param(MAX_PRIME_64 - 1, 1, id="add_wrap_exact_q"),
    pytest.param(MAX_PRIME_64 - 2, 3, id="add_wrap_over_q"),
    pytest.param(1, MAX_PRIME_64 - 1, id="sub_negative_wrap"),
    pytest.param(MAX_PRIME_64 - 1, MAX_PRIME_64 - 1, id="overflow_add_full"),
    pytest.param(MAX_PRIME_64 - 2, MAX_PRIME_64 - 2, id="overflow_add_near_full"),
    pytest.param(MAX_PRIME_64 // 2, MAX_PRIME_64 // 3, id="mid_values"),
    pytest.param(MAX_PRIME_64 - 123456789, MAX_PRIME_64 - 987654321, id="high_values"),
]

EDGE_CASES_128 = [
    pytest.param(0, 0, id="zero_zero"),
    pytest.param(1, 2, id="a_lt_b_small"),
    pytest.param(2, 1, id="a_gt_b_small"),
    pytest.param(MAX_PRIME_128 - 1, 1, id="add_wrap_exact_q"),
    pytest.param(MAX_PRIME_128 - 2, 3, id="add_wrap_over_q"),
    pytest.param(1, MAX_PRIME_128 - 1, id="sub_negative_wrap"),
    pytest.param(MAX_PRIME_128 - 1, MAX_PRIME_128 - 1, id="overflow_add_full"),
    pytest.param(MAX_PRIME_128 - 2, MAX_PRIME_128 - 2, id="overflow_add_near_full"),
    pytest.param(MAX_PRIME_128 // 2, MAX_PRIME_128 // 3, id="mid_values"),
    pytest.param(
        MAX_PRIME_128 - 123456789012345678901234567890,
        MAX_PRIME_128 - 98765432109876543210987654321,
        id="high_values",
    ),
]


@pytest.mark.parametrize("a,b", EDGE_CASES_32)
def test_mod_add_32bit_edge_cases(a, b):
    q = MAX_PRIME_32
    expected = ((int(a) % q) + (int(b) % q)) % q
    got = _eval_mod_op(student.mod_add_32, a, b, q, input_dtype=jnp.uint32)
    assert got == expected


@pytest.mark.parametrize("a,b", EDGE_CASES_32)
def test_mod_sub_32bit_edge_cases(a, b):
    q = MAX_PRIME_32
    expected = ((int(a) % q) - (int(b) % q)) % q
    got = _eval_mod_op(student.mod_sub_32, a, b, q, input_dtype=jnp.uint32)
    assert got == expected


@pytest.mark.parametrize("a,b", EDGE_CASES_32)
def test_mod_mul_32bit_edge_cases(a, b):
    q = MAX_PRIME_32
    expected = ((int(a) % q) * (int(b) % q)) % q
    got = _eval_mod_op(student.mod_mul_32, a, b, q, input_dtype=jnp.uint32)
    assert got == expected


@pytest.mark.parametrize("a,b", EDGE_CASES_64)
def test_mod_add_64bit_edge_cases(a, b):
    q = MAX_PRIME_64
    expected = ((int(a) % q) + (int(b) % q)) % q
    got = _eval_mod_op(student.mod_add_64, a, b, q, input_dtype=jnp.uint64)
    assert got == expected


@pytest.mark.parametrize("a,b", EDGE_CASES_64)
def test_mod_sub_64bit_edge_cases(a, b):
    q = MAX_PRIME_64
    expected = ((int(a) % q) - (int(b) % q)) % q
    got = _eval_mod_op(student.mod_sub_64, a, b, q, input_dtype=jnp.uint64)
    assert got == expected


@pytest.mark.parametrize("a,b", EDGE_CASES_64)
def test_mod_mul_64bit_edge_cases(a, b):
    q = MAX_PRIME_64
    expected = ((int(a) % q) * (int(b) % q)) % q
    got = _eval_mod_op(student.mod_mul_64, a, b, q, input_dtype=jnp.uint64)
    assert got == expected


@pytest.mark.parametrize("a,b", EDGE_CASES_128)
def test_mod_add_128bit_edge_cases(a, b):
    q = MAX_PRIME_128
    expected = ((int(a) % q) + (int(b) % q)) % q
    got = _eval_mod_op(student.mod_add_128, a, b, q)
    assert got == expected


@pytest.mark.parametrize("a,b", EDGE_CASES_128)
def test_mod_sub_128bit_edge_cases(a, b):
    q = MAX_PRIME_128
    expected = ((int(a) % q) - (int(b) % q)) % q
    got = _eval_mod_op(student.mod_sub_128, a, b, q)
    assert got == expected


@pytest.mark.parametrize("a,b", EDGE_CASES_128)
def test_mod_mul_128bit_edge_cases(a, b):
    q = MAX_PRIME_128
    expected = ((int(a) % q) * (int(b) % q)) % q
    got = _eval_mod_op(student.mod_mul_128, a, b, q)
    assert got == expected


def test_sumcheck_matches_expected_outputs(case_expression):
    """One pytest item per (case, expression) selected by CLI filters."""
    case, expression = case_expression
    if case is None:
        pytest.skip("No selected case/expression to run")

    tests_dir = Path(__file__).resolve().parent

    case_id = case["id"]
    expr_id = expression_to_id(expression)
    bit_width = int(case.get("prime_bits", 32))
    q_int = int(case["q"])
    q_arg = _to_jax_q(q_int)
    all_challenges = [int(x) % q_int for x in case["challenges"]]
    num_rounds = len(all_challenges)
    prover_challenges = all_challenges[:-1]

    eval_tables = _to_jax_tables(load_tables(case, tests_dir=tests_dir), q_int)
    prover_challenges = _to_jax_challenges(prover_challenges, q_int)

    expected_prover_len = len(all_challenges) - 1
    assert len(prover_challenges) == expected_prover_len, (
        f"{case_id}: sumcheck input challenges must exclude the final "
        f"verifier challenge; expected {expected_prover_len}, "
        f"got {len(prover_challenges)}"
    )

    exp = expected_entry(case, expression, q=q_int)
    exp_rounds = exp["round_evals"]
    exp_final = exp["final_eval"]
    exp_claim0 = exp["claim0"]

    got = student.sumcheck(
        eval_tables,
        q=q_arg,
        expression=expression,
        challenges=prover_challenges,
        num_rounds=num_rounds,
        bit_width=bit_width,
    )
    got_claim0, got_rounds = _extract_claim0_and_round_evals(
        got,
        q_int,
        case_id=case_id,
        expr_id=expr_id,
    )
    assert int(got_claim0) % q_int == int(exp_claim0) % q_int, (
        f"{case_id}/{expr_id}: claim0 mismatch"
    )

    assert len(got_rounds) == len(exp_rounds), (
        f"{case_id}/{expr_id}: expected {len(exp_rounds)} rounds, "
        f"got {len(got_rounds)}"
    )
    assert got_rounds == exp_rounds, (
        f"{case_id}/{expr_id}: per-round mismatch"
    )

    ok, final_claim = _verifier_consistency_check(
        got_rounds,
        got_claim0,
        all_challenges,
        q_int,
    )
    assert ok, f"{case_id}/{expr_id}: verifier consistency check failed"
    assert int(final_claim) % q_int == int(exp_final) % q_int, (
        f"{case_id}/{expr_id}: verifier final claim mismatch"
    )


def test_jit_matches_eager_on_compulsory_tracks(filtered_cases):
    """For 32/64-bit cases, jitted sumcheck must match eager output."""
    if not filtered_cases:
        pytest.skip(
            "No selected cases found in tests/data. "
            "Ensure release case data is present."
        )

    tests_dir = Path(__file__).resolve().parent

    for case in filtered_cases:
        bits = int(case.get("prime_bits", 0))
        if bits not in (32, 64):
            continue

        q_int = int(case["q"])
        q_arg = _to_jax_q(q_int)
        all_challenges = [int(x) % q_int for x in case["challenges"]]
        num_rounds = len(all_challenges)
        prover_challenges = all_challenges[:-1]
        eval_tables = load_tables(case, tests_dir=tests_dir)
        eval_tables = _to_jax_tables(eval_tables, q_int)
        prover_challenges = _to_jax_challenges(prover_challenges, q_int)

        expected_prover_len = len(all_challenges) - 1
        if len(prover_challenges) != expected_prover_len:
            pytest.fail(
                f"{case['id']}: sumcheck input challenges must exclude the final "
                f"verifier challenge; expected {expected_prover_len}, "
                f"got {len(prover_challenges)}"
            )

        expressions = iter_case_expressions(case)
        if not expressions:
            continue

        expression = expressions[0]

        eager = student.sumcheck(
            eval_tables,
            q=q_arg,
            expression=expression,
            challenges=prover_challenges,
            num_rounds=num_rounds,
            bit_width=int(case.get("prime_bits", 32)),
        )

        fn = jax.jit(
            lambda tables, rs: student.sumcheck(
                tables,
                q=q_arg,
                expression=expression,
                challenges=rs,
                num_rounds=num_rounds,
                bit_width=int(case.get("prime_bits", 32)),
            )
        )
        jitted = fn(eval_tables, prover_challenges)
        eager_claim0, eager_rounds = _extract_claim0_and_round_evals(
            eager,
            q_int,
            case_id=case["id"],
            expr_id=expression_to_id(expression),
        )
        jitted_claim0, jitted_rounds = _extract_claim0_and_round_evals(
            jitted,
            q_int,
            case_id=case["id"],
            expr_id=expression_to_id(expression),
        )

        assert eager_claim0 == jitted_claim0
        assert eager_rounds == jitted_rounds

        # Keep at least one compulsory case check; no need to repeat all.
        return

    pytest.skip("No 32/64-bit cases available for JIT consistency test")
