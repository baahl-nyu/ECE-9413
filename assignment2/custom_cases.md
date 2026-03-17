# Custom Cases Guide

This guide explains how to generate, check, and benchmark your own SumCheck
cases using `scripts/custom_cases.py`.

## 1. Why Use Custom Cases

Use custom cases to debug and profile outside the required public tracks.
Typical use cases:

- test smaller/larger `num-vars` than the public suite,
- test one expression repeatedly,
- test expressions with custom variable names.

## 2. Generate a Custom Case

Basic example:

```bash
uv run python scripts/custom_cases.py generate --num-vars 18 --expression "a*b" --bits 32 --seed 0
```

Generate with a custom case id:

```bash
uv run python scripts/custom_cases.py generate --num-vars 18 --expression "a*b" --bits 32 --seed 0 --case-id my_vars18_ab
```

Generate with custom variables:

```bash
uv run python scripts/custom_cases.py generate --num-vars 22 --expression "a*b*c*d*e*g*h" --vars "a,b,c,d,e,g,h" --bits 32 --seed 1
```

If you pass `--q` explicitly, use a **prime** modulus (the verifier interpolation
logic assumes field arithmetic).

Default case-id format (if `--case-id` is omitted):

- `custom_v<num_vars>_<bits>b_<expr_tag>_s<seed>`

Generated files are written under `tests/data/custom` by default:

- `<case_id>.npz` for tables
- `<case_id>_meta.json` for metadata (expression, modulus, challenges, file refs)

**Note**: For custom cases, an expression is **required** (unlike for the already provided test cases)

## 3. Check Your Implementation on a Custom Case

```bash
uv run python scripts/custom_cases.py check --case-id custom_v18_32b_a_b_s0
```

Or for a custom-named case:

```bash
uv run python scripts/custom_cases.py check --case-id my_vars18_ab
```

## 4. Benchmark a Custom Case

```bash
uv run python scripts/custom_cases.py bench --case-id custom_v18_32b_a_b_s0 --runs 8 --warmup 3 --show-invocation-times
```

Or for a custom-named case:

```bash
uv run python scripts/custom_cases.py bench --case-id my_vars18_ab --runs 8 --warmup 3 --show-invocation-times
```

## 5. How consistency is checked in `custom_cases.py`

In each round $i$, the verifier first checks consistency against the previous
claim:

$$
g_i(0) + g_i(1) = \mathrm{claim}_{i-1}.
$$

Then, given round evaluations `y_j = g_i(j)` for `j = 0..d`, the verifier
computes the next claim by interpolation:

$$
\mathrm{claim}_i = g_i(r_i)
$$

Ignoring the round index:

$$
g(r) = \sum_{j=0}^{d} y_j \cdot \ell_j(r),
\qquad
\ell_j(r)=\prod_{m = 0; m\neq j}^{d}\frac{r-m}{j-m}.
$$

In the example shown in `sumcheck_intro.md`, we are working with a degree-$1$ polynomial, so $d=1$. In this case, $g(r)$ simplifies to:

$$
g(r) = y_0\cdot (1-r) + y_1\cdot (r) = (y_1 - y_0)\cdot r + y_0 = (g(1) - g(0)) \cdot r + g(0)  \pmod q
$$

This is identical to the MLE Update formula:

$$
g(r) = \mathrm{mle\_update}(g(0), g(1), r)
= (g(1)-g(0)) \cdot r + g(0) \pmod q.
$$

So the MLE update formula is exactly the degree-$1$ Lagrange interpolation case. For higher-degree polynomials, the $\mathrm{claim}_i$ calculation is more complicated.

In this repo, this interpolation is already implemented in:

- `scripts/custom_cases.py` -> `_lagrange_eval_at`

And used by:

- `_verifier_check_and_update_claim`
- `_verifier_sumcheck`

So you do **not** need to implement interpolation in your student code just to
use custom-case checking; the script handles verifier-side interpolation to check
consistency.

After the transcript consistency checks, `check` also performs the final oracle
check:

$$
\mathrm{claim}_n \stackrel{?}{=} f(r_1,\ldots,r_n)
$$

where $f(r_1,\ldots,r_n)$ is computed directly from the generated tables and the
full challenge list.

Important: custom `generate` does **not** compute/store expected prover
transcripts. The `check` command runs your `student.sumcheck` and verifies that
its `(claim0, round_evals)` transcript is verifier-consistent.

## 6. Related Notes

- `generate` creates starting tables + challenges only (no expected prover transcript).
- `check` verifies your `student.sumcheck` output with verifier consistency checks
  and a final oracle check.
- `bench` times your `student.sumcheck` on the selected custom case.
