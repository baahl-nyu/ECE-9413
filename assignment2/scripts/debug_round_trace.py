#!/usr/bin/env python3
"""Debug vars4 round-table folds using provided.expression_round_trace.

This helper is for manual debugging. It is not used by pytest/benchmark.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import provided


def _mle_update_py(zero_eval: int, one_eval: int, target_eval: int, q: int) -> int:
    q = int(q)
    zero_eval = int(zero_eval) % q
    one_eval = int(one_eval) % q
    target_eval = int(target_eval) % q
    return (((one_eval - zero_eval) % q) * target_eval + zero_eval) % q


def _challenge_for_round(trace: dict, round_idx: int) -> tuple[int, str]:
    prover_challenges = trace["challenges"]
    if round_idx < len(prover_challenges):
        return int(prover_challenges[round_idx]), "prover"
    if round_idx == len(prover_challenges):
        return int(trace["verifier_final_challenge"]), "verifier-final"
    raise IndexError(
        f"round_idx={round_idx} out of range for {len(trace['round_tables']) - 1} folds"
    )


def _check_all(trace: dict, *, include_verifier_round: bool) -> int:
    q = int(trace["q"])
    vars_in_case = list(trace["starting_tables"].keys())
    total_folds = len(trace["round_tables"]) - 1
    prover_folds = len(trace["challenges"])
    max_round = total_folds if include_verifier_round else prover_folds

    checked = 0
    mismatches = []

    for round_idx in range(max_round):
        challenge, challenge_kind = _challenge_for_round(trace, round_idx)
        for var in vars_in_case:
            current = trace["round_tables"][round_idx][var]
            nxt = trace["round_tables"][round_idx + 1][var]
            for pos in range(len(nxt)):
                zero_eval = current[2 * pos]
                one_eval = current[2 * pos + 1]
                expected = int(nxt[pos]) % q
                computed = _mle_update_py(zero_eval, one_eval, challenge, q)
                checked += 1
                if computed != expected:
                    mismatches.append(
                        {
                            "round": round_idx,
                            "challenge_kind": challenge_kind,
                            "var": var,
                            "pos": pos,
                            "zero_eval": int(zero_eval),
                            "one_eval": int(one_eval),
                            "challenge": int(challenge),
                            "expected": expected,
                            "computed": computed,
                        }
                    )

    if mismatches:
        first = mismatches[0]
        print(
            "[check-all] FAIL: "
            f"{len(mismatches)} mismatches out of {checked} checked updates"
        )
        print(
            "[check-all] first mismatch: "
            f"round={first['round']} ({first['challenge_kind']}) "
            f"var={first['var']} pos={first['pos']} "
            f"zero={first['zero_eval']} one={first['one_eval']} "
            f"r={first['challenge']} expected={first['expected']} "
            f"computed={first['computed']}"
        )
        return 1

    print(
        "[check-all] PASS: "
        f"checked {checked} updates across {max_round} rounds "
        f"(include_verifier_round={include_verifier_round})"
    )
    return 0


def _inspect_one(trace: dict, *, round_idx: int, var: str, pos: int) -> int:
    q = int(trace["q"])
    if var not in trace["starting_tables"]:
        print(f"[inspect] unknown variable {var!r} for selected case")
        return 1

    total_folds = len(trace["round_tables"]) - 1
    if round_idx < 0 or round_idx >= total_folds:
        print(f"[inspect] round must be in [0, {total_folds - 1}], got {round_idx}")
        return 1

    challenge, challenge_kind = _challenge_for_round(trace, round_idx)
    current = trace["round_tables"][round_idx][var]
    nxt = trace["round_tables"][round_idx + 1][var]

    if pos < 0 or pos >= len(nxt):
        print(f"[inspect] pos must be in [0, {len(nxt) - 1}], got {pos}")
        return 1

    zero_eval = int(current[2 * pos]) % q
    one_eval = int(current[2 * pos + 1]) % q
    expected = int(nxt[pos]) % q
    computed = _mle_update_py(zero_eval, one_eval, challenge, q)

    print(
        f"[inspect] round={round_idx} ({challenge_kind}) var={var} pos={pos} q={q}"
    )
    print(
        f"[inspect] zero={zero_eval} one={one_eval} challenge={int(challenge) % q}"
    )
    print(f"[inspect] expected_next={expected} computed_next={computed}")
    print(f"[inspect] match={computed == expected}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect and validate vars4 round-table folds from expression_round_trace."
    )
    parser.add_argument(
        "--expr-index",
        type=int,
        default=0,
        help="Expression index into provided.EXPRESSIONS (default: 0)",
    )
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="Optional case id; use a vars4 case for round-table debugging",
    )
    parser.add_argument("--var", type=str, default="a", help="Variable name to inspect")
    parser.add_argument(
        "--round",
        type=int,
        default=0,
        help="Round index to inspect (default: 0)",
    )
    parser.add_argument(
        "--pos",
        type=int,
        default=0,
        help="Pair index within the selected round/variable table (default: 0)",
    )
    parser.add_argument(
        "--check-all-prover",
        action="store_true",
        help="Verify all student-relevant (prover) table updates in this trace",
    )
    parser.add_argument(
        "--check-all-including-verifier",
        action="store_true",
        help="Also verify the final verifier-only fold",
    )
    args = parser.parse_args()

    trace = provided.expression_round_trace(args.expr_index, case_id=args.case_id)
    print(
        "[trace] "
        f"case_id={trace['case_id']} expr={trace['expression']} q={trace['q']} "
        f"num_rounds={trace['num_rounds']} prover_challenges={len(trace['challenges'])}"
    )
    print(f"[trace] prover_challenges={trace['challenges']}")
    print(f"[trace] verifier_final_challenge={trace['verifier_final_challenge']}")

    rc = _inspect_one(trace, round_idx=args.round, var=args.var, pos=args.pos)
    if rc != 0:
        return rc

    if args.check_all_prover:
        rc = _check_all(trace, include_verifier_round=False)
        if rc != 0:
            return rc

    if args.check_all_including_verifier:
        rc = _check_all(trace, include_verifier_round=True)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
