"""Student-facing constants and helpers for Assignment 2 (sumcheck)."""

from __future__ import annotations

from pathlib import Path

VARIABLE_NAMES = ("a", "b", "c", "d", "e", "g")

# Expression format: list[list[str]]
# Each inner list is a multiplicative term. The outer list is additive terms.
# Examples:
#   a*b       -> [["a", "b"]]
#   a*b + c   -> [["a", "b"], ["c"]]
EXPRESSIONS = (
    [["a"]],
    [["a", "b"]],
    [["a", "b"], ["c"]],
    [["a", "b", "c"]],
    [["a", "a", "b", "b", "c"]],
    [["a", "b", "c"], ["d", "e"]],
    [["a", "b", "c", "g"], ["d", "e", "g"]],
)


def _expression_id(expression):
    return " + ".join("*".join(term) for term in expression)


def _load_round_tables(case, *, q: int):
    if "round_tables_file" not in case:
        raise ValueError(
            f"Case {case['id']} is missing 'round_tables_file'. "
            "This is expected for vars20 cases (round tables omitted for size)."
        )

    import numpy as np

    tests_dir = Path(__file__).resolve().parent / "tests"
    tables_path = (tests_dir / case["round_tables_file"]).resolve()
    if not tables_path.exists():
        raise FileNotFoundError(f"Case round tables file not found: {tables_path}")

    expected_rounds = len(case["challenges"])
    round_tables = []
    with np.load(tables_path, allow_pickle=True) as data:
        for round_idx in range(expected_rounds + 1):
            tables = {}
            for name in VARIABLE_NAMES:
                key = f"r{round_idx}_{name}"
                if key not in data:
                    raise ValueError(
                        f"Case {case['id']} file {tables_path} missing key '{key}'"
                    )
                tables[name] = [int(v) % q for v in data[key].tolist()]
            round_tables.append(tables)
    return round_tables


def _expected_from_case(case, expression, *, q: int):
    def _normalize_rounds(raw_rounds):
        if raw_rounds and isinstance(raw_rounds[0], list):
            return [[int(v) % q for v in row] for row in raw_rounds]
        return [int(v) % q for v in raw_rounds]

    expected = case.get("expected", {})

    if isinstance(expected, list):
        for item in expected:
            if not isinstance(item, dict) or "expression" not in item:
                continue
            if item["expression"] != expression:
                continue

            rounds = _normalize_rounds(item.get("round_evals", []))
            default_final = (
                rounds[-1]
                if rounds and not isinstance(rounds[-1], list)
                else 0
            )
            final_eval = int(item.get("final_eval", default_final)) % q
            return rounds, final_eval

        raise KeyError(
            f"Case {case['id']} has no expected output for expression {_expression_id(expression)!r}"
        )

    if isinstance(expected, dict):
        expr_id = _expression_id(expression)
        if expr_id not in expected:
            raise KeyError(
                f"Case {case['id']} has no expected output for expression {expr_id!r}"
            )
        raw = expected[expr_id]
        if isinstance(raw, dict):
            rounds = _normalize_rounds(raw.get("round_evals", []))
            default_final = (
                rounds[-1]
                if rounds and not isinstance(rounds[-1], list)
                else 0
            )
            final_eval = int(raw.get("final_eval", default_final)) % q
            return rounds, final_eval

        rounds = _normalize_rounds(raw)
        return rounds, (rounds[-1] if rounds else 0)

    raise ValueError(f"Case {case['id']} expected must be a list or dict")


def expression_round_trace(expression_index: int, *, case_id: str | None = None):
    """
    Return one expression trace from generated ground-truth public cases.

    Args:
        expression_index: Index into ``EXPRESSIONS`` (for example, ``2`` for
            ``[["a", "b"], ["c"]]``).
        case_id: Optional case id from ``tests/data/*/*_meta.json``. If omitted,
            the first vars4 case with round-table data is used.

    Returns:
        dict with:
            expression_index: selected expression index
            expression: polynomial expression list[list[str]]
            case_id: selected case id
            q: prime modulus
            num_rounds: total number of sumcheck rounds for this case
            challenges: prover challenge list (all but the final verifier check
                challenge)
            verifier_final_challenge: last challenge used only by verifier final
                point check
            starting_tables: round-0 variable tables
            round_tables: list of per-round variable tables (round 0 to final)
            expected_round_evals: expected per-round evaluation vectors from
                ground-truth
            final_eval: final evaluation in the folded point
    """
    if expression_index < 0 or expression_index >= len(EXPRESSIONS):
        raise IndexError(
            f"expression_index out of range: {expression_index} "
            f"(valid: 0..{len(EXPRESSIONS) - 1})"
        )

    try:
        from tests.data_loader import discover_cases
    except ImportError as exc:
        raise RuntimeError(
            "Could not import tests.data_loader. "
            "Run from the assignment2 repository root."
        ) from exc

    expression = [list(term) for term in EXPRESSIONS[expression_index]]
    cases = discover_cases()
    if not cases:
        raise RuntimeError(
            "No test cases found in tests/data. "
            "Ensure the assignment release data files are present."
        )

    if case_id is None:
        case = None
        for candidate in cases:
            if "round_tables_file" not in candidate:
                continue
            q_candidate = int(candidate["q"])
            try:
                _expected_from_case(candidate, expression, q=q_candidate)
            except KeyError:
                continue
            case = candidate
            break

        if case is None:
            raise RuntimeError(
                "No vars4 case with round-table data found for this expression."
            )
    else:
        matches = [c for c in cases if c.get("id") == case_id]
        if not matches:
            available = [c.get("id", "<missing-id>") for c in cases]
            raise KeyError(f"Unknown case_id {case_id!r}. Available: {available}")
        case = matches[0]
        if "round_tables_file" not in case:
            raise ValueError(
                f"Case {case['id']} has no round tables. "
                "Use a vars4 case id."
            )

    q = int(case["q"])
    all_challenges = [int(v) % q for v in case["challenges"]]
    if not all_challenges:
        raise ValueError(f"Case {case['id']} has no challenges")

    challenges = all_challenges[:-1]
    verifier_final_challenge = all_challenges[-1]
    expected_round_evals, final_eval = _expected_from_case(case, expression, q=q)
    round_tables = _load_round_tables(case, q=q)

    return {
        "expression_index": int(expression_index),
        "expression": expression,
        "case_id": case["id"],
        "q": q,
        "num_rounds": len(all_challenges),
        "challenges": challenges,
        "verifier_final_challenge": verifier_final_challenge,
        "starting_tables": round_tables[0],
        "round_tables": round_tables,
        "expected_round_evals": expected_round_evals,
        "final_eval": final_eval,
    }
