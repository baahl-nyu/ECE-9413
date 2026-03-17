from __future__ import annotations

import pytest

from sumcheck_utils import expression_to_id
from tests.case_utils import iter_case_expressions
from tests.case_selection import select_cases
from tests.data_loader import discover_cases

_SUMCHECK_ITEM_RESULTS = []
_OTHER_ITEM_RESULTS = []


def pytest_sessionstart(session):
    # Per-item outcomes for parameterized correctness test.
    _SUMCHECK_ITEM_RESULTS.clear()
    _OTHER_ITEM_RESULTS.clear()


def pytest_runtest_logreport(report):
    if report.when != "call":
        return
    needle = "test_sumcheck_matches_expected_outputs["

    outcome = report.outcome.upper()
    message = ""
    if report.failed:
        message = str(report.longrepr).splitlines()[-1].strip()
    if needle in report.nodeid:
        case_expr = report.nodeid.split(needle, 1)[1].rsplit("]", 1)[0]
        _SUMCHECK_ITEM_RESULTS.append((outcome, case_expr, message))
    else:
        _OTHER_ITEM_RESULTS.append((outcome, report.nodeid, message))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    results = list(_SUMCHECK_ITEM_RESULTS)
    if results:
        passed = sum(1 for outcome, _, _ in results if outcome == "PASSED")
        failed = sum(1 for outcome, _, _ in results if outcome == "FAILED")
        skipped = sum(1 for outcome, _, _ in results if outcome == "SKIPPED")

        terminalreporter.write_sep("=", "Sumcheck Case/Expression Summary")
        terminalreporter.write_line(
            f"pass={passed} fail={failed} skip={skipped} total={len(results)}"
        )
        for outcome, case_expr, message in results:
            if message:
                terminalreporter.write_line(f"{outcome:6} {case_expr} :: {message}")
            else:
                terminalreporter.write_line(f"{outcome:6} {case_expr}")

    other = list(_OTHER_ITEM_RESULTS)
    if other:
        passed = sum(1 for outcome, _, _ in other if outcome == "PASSED")
        failed = sum(1 for outcome, _, _ in other if outcome == "FAILED")
        skipped = sum(1 for outcome, _, _ in other if outcome == "SKIPPED")
        terminalreporter.write_sep("=", "Other Test Summary")
        terminalreporter.write_line(
            f"pass={passed} fail={failed} skip={skipped} total={len(other)}"
        )
        for outcome, nodeid, message in other:
            if message:
                terminalreporter.write_line(f"{outcome:6} {nodeid} :: {message}")
            else:
                terminalreporter.write_line(f"{outcome:6} {nodeid}")


def _collect_requested_bits(pytestconfig):
    requested_bits = [str(v) for v in (pytestconfig.getoption("bits") or [])]
    if pytestconfig.getoption("all_32"):
        requested_bits.append("32")
    if pytestconfig.getoption("all_64"):
        requested_bits.append("64")
    return requested_bits


def _explicit_64_selected(case_ids, requested_bits):
    if any(str(v) == "64" for v in requested_bits):
        return True
    return any("_case64_" in str(case_id) for case_id in (case_ids or []))


def _selected_cases(pytestconfig):
    requested_ids = pytestconfig.getoption("case_id") or []
    requested_bits = _collect_requested_bits(pytestconfig)
    requested_num_vars = pytestconfig.getoption("num_vars") or []
    requested_expr_ids = pytestconfig.getoption("expr_id") or []
    explicit_64_requested = _explicit_64_selected(requested_ids, requested_bits)
    enable_challenge32 = bool(pytestconfig.getoption("enable_challenge32"))
    enable_core64 = bool(pytestconfig.getoption("enable_core64") or explicit_64_requested)
    enable_challenge64 = bool(pytestconfig.getoption("enable_challenge64"))

    all_cases = discover_cases()
    return select_cases(
        all_cases,
        case_ids=requested_ids,
        bits=requested_bits,
        num_vars=requested_num_vars,
        expr_ids=requested_expr_ids,
        enable_challenge32=enable_challenge32,
        enable_core64=enable_core64,
        enable_challenge64=enable_challenge64,
    )


def pytest_addoption(parser):
    parser.addoption(
        "--case-id",
        action="append",
        default=[],
        help="Only run listed case ids (repeatable)",
    )
    parser.addoption(
        "--bits",
        action="append",
        default=[],
        help="Only run case prime bit sizes: 32, 64 (repeatable)",
    )
    parser.addoption(
        "--num-vars",
        action="append",
        default=[],
        help="Only run listed variable-count tracks (e.g. 4, 16, 20), repeatable",
    )
    parser.addoption(
        "--all-32",
        action="store_true",
        help="Select all 32-bit cases (equivalent to adding --bits 32).",
    )
    parser.addoption(
        "--all-64",
        action="store_true",
        help="Select all 64-bit cases (equivalent to adding --bits 64).",
    )
    parser.addoption(
        "--expr-id",
        action="append",
        default=[],
        help="Only run listed expression ids (repeatable), e.g. a*b + c",
    )
    parser.addoption(
        "--enable-challenge32",
        action="store_true",
        help="Include challenge expressions on 32-bit cases",
    )
    parser.addoption(
        "--enable-core64",
        action="store_true",
        help="Include core expressions on 64-bit cases",
    )
    parser.addoption(
        "--enable-challenge64",
        action="store_true",
        help="Include challenge expressions on 64-bit cases",
    )
    parser.addoption(
        "--include-128",
        action="store_true",
        help="Deprecated compatibility flag (128-bit cases are not selected here).",
    )


@pytest.fixture(scope="session")
def filtered_cases(pytestconfig):
    return _selected_cases(pytestconfig)


@pytest.fixture(scope="session")
def enable_64bit_checks(pytestconfig):
    requested_ids = pytestconfig.getoption("case_id") or []
    requested_bits = _collect_requested_bits(pytestconfig)
    explicit_64_requested = _explicit_64_selected(requested_ids, requested_bits)
    return bool(
        pytestconfig.getoption("enable_core64")
        or pytestconfig.getoption("enable_challenge64")
        or explicit_64_requested
    )


@pytest.fixture(scope="session")
def enable_128bit_checks(pytestconfig):
    return bool(pytestconfig.getoption("include_128"))


def pytest_generate_tests(metafunc):
    if "case_expression" not in metafunc.fixturenames:
        return

    selected_cases = _selected_cases(metafunc.config)
    params = []
    for case in selected_cases:
        case_id = str(case.get("id", ""))
        expressions = iter_case_expressions(case)
        if not expressions:
            params.append(
                pytest.param(
                    (None, None),
                    id=f"{case_id}::no-expressions",
                    marks=pytest.mark.skip(
                        reason=f"Case {case_id} has no expected expressions"
                    ),
                )
            )
            continue
        for expression in expressions:
            expr_id = expression_to_id(expression)
            params.append(
                pytest.param((case, expression), id=f"{case_id}::{expr_id}")
            )

    if not params:
        params.append(
            pytest.param(
                (None, None),
                id="no-selected-cases",
                marks=pytest.mark.skip(
                    reason=(
                        "No selected cases found in tests/data. "
                        "Ensure release case data is present."
                    )
                ),
            )
        )

    metafunc.parametrize("case_expression", params)


def pytest_collection_modifyitems(config, items):
    """Hide optional primitive tracks unless their flags are enabled."""
    requested_ids = config.getoption("case_id") or []
    requested_bits = _collect_requested_bits(config)
    explicit_64_requested = _explicit_64_selected(requested_ids, requested_bits)
    enable_64 = bool(
        config.getoption("enable_core64")
        or config.getoption("enable_challenge64")
        or explicit_64_requested
    )
    enable_128 = bool(config.getoption("include_128"))

    keep = []
    for item in items:
        nodeid = item.nodeid
        if (
            (not enable_64)
            and "tests/test_sumcheck.py::test_mod_" in nodeid
            and "_64bit_" in nodeid
        ):
            continue
        if (
            (not enable_128)
            and "tests/test_sumcheck.py::test_mod_" in nodeid
            and "_128bit_" in nodeid
        ):
            continue
        keep.append(item)

    items[:] = keep
