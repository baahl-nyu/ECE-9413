"""Utility helpers for sumcheck tests/tooling.

This module keeps test utilities out of ``provided.py`` so the student-facing
file stays small and focused.
"""

from __future__ import annotations

import provided


def normalize_expression(expression):
    """Validate and normalize expression into tuple[tuple[str, ...], ...]."""
    if not isinstance(expression, (list, tuple)):
        raise TypeError(
            "expression must be list[list[str]] or tuple[tuple[str,...], ...]"
        )

    norm_terms = []
    for term in expression:
        if not isinstance(term, (list, tuple)) or len(term) == 0:
            raise ValueError("Each additive term must be a non-empty list of vars")

        norm_term = []
        for var in term:
            if not isinstance(var, str):
                raise TypeError("Variable names in expression must be strings")
            if var not in provided.VARIABLE_NAMES:
                raise ValueError(
                    f"Unknown variable {var!r}; expected one of {provided.VARIABLE_NAMES}"
                )
            norm_term.append(var)

        norm_terms.append(tuple(norm_term))

    if not norm_terms:
        raise ValueError("expression must contain at least one term")

    return tuple(norm_terms)


def expression_to_lists(expression):
    """Return expression in canonical list[list[str]] format."""
    norm = normalize_expression(expression)
    return [list(term) for term in norm]


def expression_to_id(expression):
    """Stable text id for expression, e.g. 'a*b + c'."""
    norm = normalize_expression(expression)
    return " + ".join("*".join(term) for term in norm)


EXPRESSION_IDS = tuple(expression_to_id(expr) for expr in provided.EXPRESSIONS)
EXPRESSION_BY_ID = {
    expr_id: expression_to_lists(expr)
    for expr_id, expr in zip(EXPRESSION_IDS, provided.EXPRESSIONS)
}


def expression_from_id(expr_id: str):
    """Convert string expression id back to list[list[str]] expression."""
    if expr_id not in EXPRESSION_BY_ID:
        raise KeyError(f"Unknown expression id: {expr_id}")
    return [list(term) for term in EXPRESSION_BY_ID[expr_id]]
