"""Shared helpers for loading and validating public sumcheck test cases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

import provided
import sumcheck_utils


def _to_int_list(values, q):
    return [int(v) % q for v in values]


def _normalize_round_evals(values, q):
    if values and isinstance(values[0], list):
        return [[int(v) % q for v in row] for row in values]
    return [int(v) % q for v in values]


def _expr_id(expression):
    return sumcheck_utils.expression_to_id(expression)


def _expr_lists(expression):
    return sumcheck_utils.expression_to_lists(expression)


def load_tables(case, *, tests_dir: Path):
    """Load case tables from inline payload or external .npz file."""
    q = int(case["q"])

    if "tables" in case and "table_file" in case:
        raise ValueError(f"Case {case['id']} has both tables and table_file")

    if "tables" in case:
        tables = case["tables"]
        out = {}
        for name in provided.VARIABLE_NAMES:
            if name not in tables:
                raise ValueError(f"Case {case['id']} missing table '{name}'")
            out[name] = _to_int_list(tables[name], q)
        return _validate_table_lengths(case["id"], out)

    if "table_file" in case:
        table_path = (tests_dir / case["table_file"]).resolve()
        if not table_path.exists():
            raise FileNotFoundError(f"Case table file not found: {table_path}")

        with np.load(table_path, allow_pickle=True) as data:
            out = {}
            for name in provided.VARIABLE_NAMES:
                if name not in data:
                    raise ValueError(
                        f"Case {case['id']} file {table_path} missing key '{name}'"
                    )
                raw = data[name].tolist()
                out[name] = _to_int_list(raw, q)
        return _validate_table_lengths(case["id"], out)

    raise ValueError(f"Case {case['id']} must provide either tables or table_file")


def _validate_table_lengths(case_id: str, tables):
    lengths = {len(v) for v in tables.values()}
    if len(lengths) != 1:
        raise ValueError(f"Case {case_id} has inconsistent table lengths")

    n = next(iter(lengths))
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Case {case_id} table length must be power of two, got {n}")

    return tables


def iter_case_expressions(case):
    """Return expressions to test for a case as list[list[str]] values."""
    expected = case.get("expected", {})
    if not expected:
        return []

    # New schema: expected is a list of entries with an "expression" field.
    if isinstance(expected, list):
        out = []
        seen = set()
        for item in expected:
            if not isinstance(item, dict) or "expression" not in item:
                raise ValueError(
                    f"Case {case['id']} expected list entries need an 'expression' field"
                )
            expr = _expr_lists(item["expression"])
            expr_id = _expr_id(expr)
            if expr_id not in seen:
                out.append(expr)
                seen.add(expr_id)
        return out

    # Backward compatibility: expected as dict keyed by expression id strings.
    if isinstance(expected, dict):
        ordered = []
        seen = set()

        for expr in provided.EXPRESSIONS:
            expr_id = _expr_id(expr)
            if expr_id in expected:
                expr_list = _expr_lists(expr)
                ordered.append(expr_list)
                seen.add(expr_id)

        for key in expected:
            if isinstance(key, str):
                if key in sumcheck_utils.EXPRESSION_BY_ID and key not in seen:
                    ordered.append(sumcheck_utils.expression_from_id(key))
                    seen.add(key)

        return ordered

    raise ValueError(f"Case {case['id']} expected must be a list or dict")

def expected_entry(case, expression, *, q):
    """Return normalized expected entry with round_evals/final_eval/claim0."""
    expected = case.get("expected", {})
    expr_id = _expr_id(expression)

    raw = None
    if isinstance(expected, list):
        for item in expected:
            if not isinstance(item, dict) or "expression" not in item:
                continue
            if _expr_id(item["expression"]) == expr_id:
                raw = item
                break
        if raw is None:
            raise KeyError(f"No expected output for expression {expr_id!r}")

    elif isinstance(expected, dict):
        if expr_id not in expected:
            raise KeyError(f"No expected output for expression {expr_id!r}")
        raw = expected[expr_id]

    else:
        raise ValueError(f"Case {case['id']} expected must be a list or dict")

    if isinstance(raw, dict):
        round_evals = _normalize_round_evals(raw.get("round_evals", []), q)
        if "final_eval" in raw:
            final_eval = int(raw["final_eval"]) % q
        elif round_evals and not isinstance(round_evals[-1], list):
            final_eval = round_evals[-1]
        else:
            final_eval = 0

        claim0 = raw.get("claim0")
        if claim0 is None:
            if round_evals and isinstance(round_evals[0], list) and len(round_evals[0]) >= 2:
                claim0 = (int(round_evals[0][0]) + int(round_evals[0][1])) % q
            else:
                claim0 = 0
        else:
            claim0 = int(claim0) % q

        return {
            "round_evals": round_evals,
            "final_eval": final_eval,
            "claim0": claim0,
        }

    round_evals = _normalize_round_evals(raw, q)
    final_eval = (
        round_evals[-1]
        if round_evals and not isinstance(round_evals[-1], list)
        else 0
    )
    claim0 = 0
    if round_evals and isinstance(round_evals[0], list) and len(round_evals[0]) >= 2:
        claim0 = (int(round_evals[0][0]) + int(round_evals[0][1])) % q

    return {
        "round_evals": round_evals,
        "final_eval": final_eval,
        "claim0": claim0,
    }
