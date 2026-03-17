"""Case/track selection helpers for pytest and benchmark harnesses."""

from __future__ import annotations

import copy

import provided
import sumcheck_utils

CORE_EXPRESSION_COUNT = 4
CORE_EXPRESSION_IDS = tuple(
    sumcheck_utils.expression_to_id(expr)
    for expr in provided.EXPRESSIONS[:CORE_EXPRESSION_COUNT]
)
CHALLENGE_EXPRESSION_IDS = tuple(
    sumcheck_utils.expression_to_id(expr)
    for expr in provided.EXPRESSIONS[CORE_EXPRESSION_COUNT:]
)


def _allowed_expression_ids(
    *,
    enable_challenge32: bool,
    enable_core64: bool,
    enable_challenge64: bool,
):
    by_bits = {
        32: set(CORE_EXPRESSION_IDS),
        64: set(),
    }

    if enable_challenge32:
        by_bits[32].update(CHALLENGE_EXPRESSION_IDS)
    if enable_core64:
        by_bits[64].update(CORE_EXPRESSION_IDS)
    if enable_challenge64:
        by_bits[64].update(CHALLENGE_EXPRESSION_IDS)

    return by_bits


def _expr_id(expression):
    return sumcheck_utils.expression_to_id(
        sumcheck_utils.expression_to_lists(expression)
    )


def _filter_expected(case, allowed_expr_ids):
    expected = case.get("expected", {})

    if isinstance(expected, list):
        filtered = []
        for item in expected:
            if not isinstance(item, dict) or "expression" not in item:
                continue
            if _expr_id(item["expression"]) in allowed_expr_ids:
                filtered.append(item)
        out = copy.copy(case)
        out["expected"] = filtered
        return out

    if isinstance(expected, dict):
        filtered = {
            expr_id: value
            for expr_id, value in expected.items()
            if expr_id in allowed_expr_ids
        }
        out = copy.copy(case)
        out["expected"] = filtered
        return out

    raise ValueError(f"Case {case.get('id', '<missing-id>')} expected must be list/dict")


def select_cases(
    cases,
    *,
    case_ids=None,
    bits=None,
    num_vars=None,
    expr_ids=None,
    enable_challenge32=False,
    enable_core64=False,
    enable_challenge64=False,
):
    requested_ids = set(case_ids or [])
    requested_bits = {int(v) for v in (bits or [])}
    requested_num_vars = {int(v) for v in (num_vars or [])}
    requested_expr_ids = set(expr_ids or [])

    allowed_by_bits = _allowed_expression_ids(
        enable_challenge32=bool(enable_challenge32),
        enable_core64=bool(enable_core64),
        enable_challenge64=bool(enable_challenge64),
    )

    selected = []
    for case in cases:
        case_id = str(case.get("id", ""))
        case_bits = int(case.get("prime_bits", 0))
        case_num_vars = int(case.get("num_vars", 0))

        if requested_ids and case_id not in requested_ids:
            continue
        if requested_bits and case_bits not in requested_bits:
            continue
        if requested_num_vars and case_num_vars not in requested_num_vars:
            continue

        allowed_expr_ids = set(allowed_by_bits.get(case_bits, set()))
        if requested_expr_ids:
            allowed_expr_ids &= requested_expr_ids
        if not allowed_expr_ids:
            continue

        filtered_case = _filter_expected(case, allowed_expr_ids)
        expected = filtered_case.get("expected", {})
        if isinstance(expected, list) and not expected:
            continue
        if isinstance(expected, dict) and not expected:
            continue

        selected.append(filtered_case)

    return selected
