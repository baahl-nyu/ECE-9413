# Assignment 2: Sumcheck in JAX

Sumcheck is an interactive protocol that allows a verifier to efficiently check the correctness of a claimed computation performed by a prover. It is a core building block in modern proof systems, where large computations are encoded as polynomials and their correctness can be verified without recomputing the entire computation.

In this assignment, you will be implementing the prover-side logic of the SumCheck protocol using JAX. This is a new assignment, so if you find any bugs or issues, **please reach out to the Course Assistants ASAP.**

## 1. Sumcheck Overview

Sumcheck is used to verify a claim about a polynomial sum over the Boolean hypercube:

$$
S = \sum_{(x_1,\dots,x_n)\in{\lbrace 0,1\rbrace}^n} f(x_1,\dots,x_n)
$$

There are two parties in SumCheck, a prover and a verifier. The protocol runs in $n$ rounds. In each round $i$:

- the prover sends evaluations of a univariate polynomial $g_i(t)$
  at points $t=0,1,\dots,d$ (where $d$ is the degree of the round polynomial),
- the verifier checks the consistency condition

$$g_i(0)+g_i(1)=\mathrm{claim}_{i-1},$$

- the verifier samples a random challenge $r_i$ and computes

$$ \mathrm{claim}_i = g_i(r_i) $$

  by interpolating from the provided evaluations.

Each round removes one Boolean variable from the remaining sum. After $n$
rounds, the original high-dimensional claim is reduced to a claim about a
single evaluation of $f$ at a random point.

In this assignment, you will be implementing the code for the **prover** steps, i.e., generating the evaluations in each SumCheck round. For a step-by-step worked example (with concrete tables, per-round updates,
and implementation hints), see `sumcheck_intro.md`.

## 2. SumCheck Resources

Read `sumcheck_intro.md` for a step-by-step protocol walkthrough with concrete round examples and hints for the assignment. There is also a powerpoint titled `sumcheck_slides.pptx` that shows visuals of the dataflow. We recommend you open this in presenter view to see the animations.

The following is recommended but optional reading to better understand SumCheck:

- [NoCap paper (MICRO '24)](https://people.csail.mit.edu/devadas/pubs/micro24_nocap.pdf) - a paper accelerating low-degree SumChecks, see Listing 1 for a high-level description of the base algorithm
- [zkSpeed paper (ISCA '25)](https://arxiv.org/pdf/2504.06211) - a paper accelerating high-degree SumChecks, see Sections 2.2, 3.1, 3.2, and Figure 1.
- [zkPHIRE paper (HPCA '26)](https://arxiv.org/pdf/2508.16738) - followup to zkSpeed that also accelerates high-degree SumChecks, see Section C and Figure 1.
- [SumCheck Protocol](https://people.cs.georgetown.edu/jthaler/sumcheck.pdf) - course material from Prof. Justin Thaler, a bit more math heavy
- [The Unreasonable Power of the Sum-Check Protocol](https://zkproof.org/2020/03/16/sum-checkprotocol/)
- [Sum-Check Protocol and Multilinear Extensions (MLEs)](https://risencrypto.github.io/Sumcheck/)

## 3. Polynomials You Will Be Tested On

You will run SumCheck on expressions built from component polynomials.

Notation clarification:

- `a`, `b`, `c`, `d`, `e`, `g` are each multilinear polynomials over the same Boolean domain.
- In code, each is provided as its own evaluation table, e.g. `eval_tables["a"]`.
- Expressions like `a*b + c` mean pointwise composition:

$$f(x) = a(x)b(x) + c(x) \pmod q.$$


### Base polynomials (compulsory)

1. `a`
2. `a*b`
3. `a*b + c`
4. `a*b*c`

### Advanced polynomials (for extra credit)

1. `a*a*b*b*c`
2. `a*b*c + d*e`
3. `a*b*c*g + d*e*g`

## 4. Polynomial Representation

Polynomials are represented as expressions using this format: `list[list[str]]`

- Outer list: additive terms
- Inner list: multiplicative factors in that term

Examples:

- `a*b` -> `[["a", "b"]]`
- `a*b + c` -> `[["a", "b"], ["c"]]`

Variable values come from `eval_tables`, keyed by variable name (for example `eval_tables["a"]`).

## 5. Required Deliverables (Must Pass)

You only edit `student.py`.

You must complete:

- Correct 32-bit primitives and 32-bit sumcheck.
- Correctness on vars4, vars16, and vars20 for base polynomials (`a`, `a*b`, `a*b + c`, `a*b*c`).
- Benchmarks on the same required scope.

Student-facing API:

- `sumcheck(...)` returns `(claim0, round_evals)`.
- Both outputs must be JAX arrays.
- `challenges` passed to student `sumcheck` exclude the final verifier-only challenge.

## 6. Setup and Platform Selection

From the assignment root:

```bash
bash scripts/setup.sh
```

After setup, select your backend if needed:

```bash
export JAX_PLATFORMS=cpu
export JAX_PLATFORMS=gpu # or cuda
export JAX_PLATFORMS=tpu
```

Notes:

- Use `gpu` or `cuda` for CUDA-backed execution.
- If your shell has another virtualenv active, use `uv run --active ...`.

## 7. Recommended Completion Strategy

Use this order to avoid long-debug cycles:

1. Run vars4 tests first.
2. Run vars16 tests second.
3. Run vars20 tests last.
4. Optionally debug first on small custom cases (`--num-vars 6`, `8`, etc.).

The first-run of JIT compile can take a while compared to steady-state timing on both CPU and GPU. Make sure your code runs on small problem sizes (`num-vars=4`) before running on larger problems.

## 8. Testcase Information

Public test data lives in `tests/data`.

- Each track has 5 cases (`index` 0 to 4).
- Tracks are grouped by variable count (`vars4`, `vars16`, `vars20`) and bit-width (32/64).
- Each case includes starting tables, challenges, and expected outputs for supported polynomials.
- You can inspect the case metadata directly in `tests/data/*/*_meta.json` if you want to see exact values and file references.

Case-id naming:

- Built-in: `v<num_vars>_case<bits>_<index>`
- Example: `v16_case32_2`
  - `num_vars=16`
  - `bits=32`
  - `index=2`

Custom case ids (`scripts/custom_cases.py generate`):

- Auto-generated (if `--case-id` is omitted):
  - `custom_v<num_vars>_<bits>b_<expr_tag>_s<seed>`
  - Example: `custom_v18_32b_a_b_s0`
- Manually named (if `--case-id` is provided):
  - Example: `--case-id my_vars18_ab`

## 9. Required Correctness Commands

Run staged checks:

```bash
uv run pytest --bits 32 --num-vars 4
uv run pytest --bits 32 --num-vars 16
uv run pytest --bits 32 --num-vars 20
```

Or run required 32-bit coverage in one command:

```bash
uv run pytest --all-32 --num-vars 4 --num-vars 16 --num-vars 20
```

Run a single polynomial/expression if you want focused debugging:

```bash
uv run pytest --bits 32 --num-vars 16 --expr-id "a*b + c"
```
You can also add `--case-id` (optional) to focus on one specific case.  
Without `--case-id`, pytest runs all selected cases for that expression.

```bash
uv run pytest --bits 32 --num-vars 16 --expr-id "a*b + c" --case-id v16_case32_2
```

## 10. Required Benchmark Commands

To run a benchmarking test, specify the bitwidth, problem size (num-vars), and how many warmup and actual runs you do. We recommend 3 warmup runs and 8 actual runs:

```bash
uv run python -m tests.benchmark --bench --bits 32 --num-vars 4 --runs 8 --warmup 3
uv run python -m tests.benchmark --bench --bits 32 --num-vars 16 --runs 8 --warmup 3
uv run python -m tests.benchmark --bench --bits 32 --num-vars 20 --runs 8 --warmup 3
```

By default, benchmark mode uses base polynomials only. Use `--show-invocation-times` if you want per-run timing data.

To benchmark one polynomial:

```bash
uv run python -m tests.benchmark --bench --bits 32 --num-vars 16 --expr "a*b" --runs 8 --warmup 3
```

## 11. vars4 Round Tables (Quick Debug View)

Per-round tables are available for vars4 cases (not for vars16/vars20).

For a quick debug snapshot, run:

```python
import provided
trace = provided.expression_round_trace(2, case_id="v4_case32_<test_number>") # test_number is one of {0, 1, 2, 3, 4}
print(f"expressions: {trace['expression']}")
print(f"challenges: {trace['challenges']}")
print(f"expected round evals: {trace['expected_round_evals']}")

for round_idx, tables in enumerate(trace["round_tables"]):
    print(f"\n=== round_tables[{round_idx}] ===")
    for var_name, values in tables.items():
        print(f"{var_name}: {values}")
```

This is optional and only intended for manual debugging.

## 12. Configuration

Enable optional polynomial/expression tracks:

- 32-bit advanced polynomials: `--enable-challenge32`
- 64-bit core: `--enable-core64` (or select 64-bit cases explicitly)
- 64-bit advanced polynomials: `--enable-challenge64`

Optional correctness examples:

```bash
uv run pytest --all-64 --enable-core64
uv run pytest --all-64 --enable-challenge64
uv run pytest --bits 32 --num-vars 20 --enable-challenge32
```

Optional benchmark examples:

```bash
uv run python -m tests.benchmark --bench --all-64 --enable-core64 --runs 8 --warmup 3
uv run python -m tests.benchmark --bench --bits 32 --num-vars 20 --enable-challenge32 --runs 8 --warmup 3
```

128-bit is optional. Primitive edge-case tests:

```bash
uv run pytest -q tests/test_sumcheck.py::test_mod_add_128bit_edge_cases --include-128
uv run pytest -q tests/test_sumcheck.py::test_mod_sub_128bit_edge_cases --include-128
uv run pytest -q tests/test_sumcheck.py::test_mod_mul_128bit_edge_cases --include-128
```

## 13. Custom Student Cases

For custom case generation/checking/benchmarking, see:

- `custom_cases.md`
- There are also notes in this readme on the math that are optional for understanding but add additional context.

## 14. Other Extra Credit Opportunities

You can additionally implement any of these and include your results in a report for extra credit.

- Implement hashing in between SumCheck rounds and see how much the performance slows down.
- Implement and optimize the 64-bit SumCheck path (`--enable-core64`, `--enable-challenge64`).
- Implement 128-bit primitives and run 128-bit SumChecks.
- Run the advanced polynomials on both CPU and GPU backends.
- Run any of the polynomials on a TPU backend.
- Run the polynomials from the zkSpeed or zkPHIRE papers.

## 15. Submission Checklist

- 32-bit correctness passed on vars4, vars16, vars20 for base expressions.
- 32-bit benchmarks collected on vars4, vars16, vars20 for base expressions.
- a report on your results inside `report.pdf` (details TBD)

