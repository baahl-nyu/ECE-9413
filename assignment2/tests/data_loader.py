"""Load public sumcheck cases directly from tests/data metadata files."""

from __future__ import annotations

import json
import re
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
DATA_DIR = THIS_DIR / "data"
_CASE_ID_RE = re.compile(r"^v(?P<num_vars>\d+)_case(?P<bits>\d+)_\d+$")


def _relative_to_tests(path: Path) -> str:
    return str(path.resolve().relative_to(THIS_DIR.resolve()))


def _load_meta_file(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Case metadata must be a JSON object: {path}")
    return payload


def _validate_case(case: dict, *, meta_path: Path):
    required = ("id", "prime_bits", "q", "challenges", "table_file", "expected")
    missing = [key for key in required if key not in case]
    if missing:
        raise ValueError(f"Case metadata missing keys {missing}: {meta_path}")

    case_id = str(case["id"])
    q = int(case["q"])
    bits = int(case["prime_bits"])
    num_vars_raw = case.get("num_vars", None)
    if num_vars_raw is None:
        m = _CASE_ID_RE.match(case_id)
        if m:
            num_vars = int(m.group("num_vars"))
        else:
            raise ValueError(
                f"Case {case_id} missing num_vars and cannot infer it from id: {meta_path}"
            )
    else:
        num_vars = int(num_vars_raw)

    challenges = case["challenges"]
    if not isinstance(challenges, list):
        raise ValueError(f"Case {case_id} challenges must be a list: {meta_path}")

    expected = case["expected"]
    if not isinstance(expected, (list, dict)):
        raise ValueError(
            f"Case {case_id} expected must be a list or dict: {meta_path}"
        )

    table_file = str(case["table_file"])
    table_path = (THIS_DIR / table_file).resolve()
    if not table_path.exists():
        raise FileNotFoundError(f"Case {case_id} table file not found: {table_path}")

    out = dict(case)
    out["id"] = case_id
    out["q"] = q
    out["prime_bits"] = bits
    out["num_vars"] = num_vars
    out["challenges"] = [int(v) % q for v in challenges]
    out["table_file"] = table_file
    out["expected"] = expected
    out["case_meta_file"] = _relative_to_tests(meta_path)

    if "round_tables_file" in out:
        round_tables_file = str(out["round_tables_file"])
        round_tables_path = (THIS_DIR / round_tables_file).resolve()
        if not round_tables_path.exists():
            raise FileNotFoundError(
                f"Case {case_id} round tables file not found: {round_tables_path}"
            )
        out["round_tables_file"] = round_tables_file

    return out


def discover_case_meta_files(*, data_dir: Path | None = None):
    root = data_dir or DATA_DIR
    if not root.exists():
        return []

    meta_files = []
    for vars_dir in sorted(root.glob("vars*")):
        if vars_dir.is_dir():
            meta_files.extend(sorted(vars_dir.glob("*_meta.json")))
    return meta_files


def discover_cases(*, data_dir: Path | None = None):
    cases = []
    for meta_path in discover_case_meta_files(data_dir=data_dir):
        raw = _load_meta_file(meta_path)
        cases.append(_validate_case(raw, meta_path=meta_path))

    cases.sort(key=lambda c: (int(c.get("prime_bits", 0)), str(c.get("id", ""))))
    return cases
