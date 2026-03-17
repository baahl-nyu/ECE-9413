# Introduction to the SumCheck Protocol

The **SumCheck protocol** is a method for verifying claims about the sum of a polynomial over all Boolean inputs. The claim is made by a party called the **prover** to another party called the **verifier**. The statement the prover wants the verifier to accept has the form

$$
S = \sum_{(x_1,\ldots,x_n)\in{\lbrace 0,1\rbrace}^n} f(x_1,\dots,x_n).
$$

That is, if we took some multivariate polynomial $f$, computed its evaluations at binary values of the variables $x_1 \ldots x_n$, and summed these evaluations, we'd expect to get some value $S$. In SumCheck, we further assume that the polynomials are **multilinear**, that is, each variable has a maximum degree of 1.

Example of multilinear polynomials: 

$x_1x_2x_3 + x_1x_3 + x_2$

$2x_1 + 3x_2 + x_2x_3$

Examples of non-multilinear polynomials:

$x_1^2x_3$

$x_1 + x_2 + x_3^3 + x_1x_2$

**In SumCheck, we only care about multilinear polynomials.** A neat consequence of this constraint is that we can represent multilinear polynomials like $f$ using a **lookup table over Boolean indices**. A multilinear polynomial over $n$ variables $x_1 \ldots x_n$ is completely determined by its values on the Boolean n-cube $\{0,1\}^n$, so we can store these values in a table of length $2^n$. We'll refer to these as multilinear extension tables (MLE tables) for the polynomial.

Now let's revisit the original formula to calculate $S$. If the goal were simply to compute $S$ once, the prover could just sum the $2^n$ values inside the lookup table representing $f$ and add them directly. However, this is necessary but not sufficient, and SumCheck goes further. 

SumCheck is an **interactive protocol** between a prover and a verifier. The prover starts from a claim about a sum over $2^n$ Boolean points and reduces it across rounds by sending a univariate polynomial $g_i(t)$ for round $i$. The verifier checks the round consistency condition

$$
g_i(0)+g_i(1)=\mathrm{claim}_{i-1},
$$

then samples a random challenge $r_i$ and updates the claim to

$$
\mathrm{claim}_i=g_i(r_i).
$$

Each round removes one Boolean variable from the remaining sum, so after $n$ rounds the original high-dimensional claim is reduced to a claim about a single evaluation of $f$ at a random point.

In practical deployments of SumCheck, the prover generates the challenges by hashing the round evaluations, so there is no per-round communication between the prover and verifier. In this assignment, **we will provide the challenges ahead of time**, so you can focus on optimizing the dataflow within a SumCheck round. However, you **must serialize** each round computation, i.e., you **cannot** run multiple rounds in parallel. 

---
# Example Protocol

We work over the field $\mathbb{F}_{17}$ and define

$$
f(x_1,x_2,x_3) = x_1 + 2x_2 + 4x_3 \pmod{17}.
$$

For this example, we index the evaluation table so that:

- $x_1$ is the **least-significant bit**
- $x_2$ is the middle bit
- $x_3$ is the **most-significant bit**

Evaluating $f$ over all Boolean inputs gives:

| idx | (x3,x2,x1) | f |
|---:|:---:|---:|
|0|(0,0,0)|0|
|1|(0,0,1)|1|
|2|(0,1,0)|2|
|3|(0,1,1)|3|
|4|(1,0,0)|4|
|5|(1,0,1)|5|
|6|(1,1,0)|6|
|7|(1,1,1)|7|

So the evaluation vector is

$$
[0,1,2,3,4,5,6,7].
$$

The claimed sum over the Boolean cube is

$$
S = 0+1+2+3+4+5+6+7 = 28 \equiv 11 \pmod{17}.
$$

Because this polynomial is defined over **3** variables, the SumCheck protocol verifies this claim over **3 rounds**, one for each variable. In round $i$, we eliminate variable $x_i$ by summing over the remaining Boolean variables while treating $x_i$ as a free variable $t$. 

We will call this resulting univariate polynomial $g_i(t)$. It must satisfy the consistency condition

$$
g_i(0) + g_i(1) = \text{claim}_{i-1}.
$$

For round $i = 1$, we set $\text{claim}_{i-1} = \text{claim}_{0} = S$ from above. Since the original sum includes both Boolean values of the chosen variable, the sum of the polynomial evaluated at $0$ and $1$ must equal the previous claim.

---

# Round 1 (Eliminate $x_1$)

We define

$$
g_1(t) = \sum_{(x_2,x_3)\in{\lbrace 0,1\rbrace}^2} f(t,x_2,x_3).
$$

Since $x_1$ is the **least-significant bit**, fixing $x_1=0$ selects the **even indices**, and fixing $x_1=1$ selects the **odd indices**.

| x1 | indices | values |
|---|---|---|
|0|0,2,4,6|0,2,4,6|
|1|1,3,5,7|1,3,5,7|

So

$$
g_1(0) = 0+2+4+6 = 12
$$

and

$$
g_1(1) = 1+3+5+7 = 16.
$$

The consistency check is

$$
g_1(0) + g_1(1) = 12 + 16 = 28 \equiv 11 \pmod{17},
$$

which matches the claimed sum $S$.

The verifier then samples a random field element $r_1$ and updates the claim to

$$
\text{claim}_1 = g_1(r_1).
$$

At this point, the Boolean variable $x_1$ has been replaced by the random field value $r_1$. Suppose the verifier samples $r_1 = 8 \in \mathbb{F}_{17}$. We now **update the table along $x_1$** using

$$
\mathrm{mle\_update}(z,o,t) = (o-z)t + z \pmod{17},
$$

with $t=r_1=8$, where:

- $z$ is the value at $x_1=0$, the even index ("z" is for zero),
- $o$ is the value at $x_1=1$, the odd index ("o" is for one).

Original table:

$$
[0,1,2,3,4,5,6,7].
$$

Pairwise updates:

- $(0,1)$: $(1-0)\cdot 8 + 0 = 8$
- $(2,3)$: $(3-2)\cdot 8 + 2 = 10$
- $(4,5)$: $(5-4)\cdot 8 + 4 = 12$
- $(6,7)$: $(7-6)\cdot 8 + 6 = 14$

So the new table (now indexed by $(x_3,x_2)$) is:

$$
[8,10,12,14].
$$

| new idx | $(x_3,x_2)$ | updated value $f(r_1,x_2,x_3)$ |
|---:|:---:|---:|
| 0 | (0,0) | 8 |
| 1 | (0,1) | 10 |
| 2 | (1,0) | 12 |
| 3 | (1,1) | 14 |

The updated claim is computed by the verifier via interpolation.
Since this round is degree 1, interpolation is exactly the MLE update:

$$
\mathrm{claim}_1 = g_1(8)
= \mathrm{mle\_update}(g_1(0), g_1(1), 8)
= (16-12)\cdot 8 + 12
= 44 \equiv 10 \pmod{17}.
$$

**What the Prover computes:**

- The round-1 polynomial evaluations: $g_1(0),\ g_1(1)$
- The table updates once the challenge is known.

**What the Verifier computes:**

- Checks round consistency: $g_1(0)+g_1(1)\stackrel{?}{=}\text{claim}_0$
- Samples random challenge: $r_1\in\mathbb{F}_{17}$. (We provide these challenges in this assignment)
- Updates claim: $\text{claim}_1=g_1(r_1)$

**Assignment note:** The claim updates and consistency checks are verifier-side, so you do **NOT** need to implement them.

---

# Round 2 (Eliminate $x_2$)

Now we define

$$
g_2(t) = \sum_{x_3\in{\lbrace 0,1\rbrace}} f(r_1,t,x_3).
$$

Since $x_2$ is now the least-significant bit in the updated 2-variable table, fixing $x_2=0$ selects even indices, and fixing $x_2=1$ selects odd indices.

Current table from Round 1:

$$
[8,10,12,14].
$$

Grouped by $x_2$:

| x2 | indices | values |
|---|---|---|
|0|0,2|8,12|
|1|1,3|10,14|

So

$$
g_2(0) = 8+12 = 20 \equiv 3 \pmod{17}
$$

and

$$
g_2(1) = 10+14 = 24 \equiv 7 \pmod{17}.
$$

Consistency check:

$$
g_2(0)+g_2(1)=3+7=10 \equiv \text{claim}_1 \pmod{17},
$$

which matches $\text{claim}_1=10$ from Round 1.

Suppose the verifier samples $r_2 = 5 \in \mathbb{F}_{17}$. We now **update the table along $x_2$** using

$$
\mathrm{mle\_update}(z,o,t) = (o-z)t + z \pmod{17},
$$

with $t=r_2=5$, where:

- $z$ is the value at $x_2=0$ (even index),
- $o$ is the value at $x_2=1$ (odd index).

Pairwise updates:

- $(8,10)$: $(10-8)\cdot 5 + 8 = 18 \equiv 1 \pmod{17}$
- $(12,14)$: $(14-12)\cdot 5 + 12 = 22 \equiv 5 \pmod{17}$

So the new table (now indexed by $x_3$ only) is:

$$
[1,5].
$$

| new idx | $(x_3)$ | updated value $f(r_1,r_2,x_3)$ |
|---:|:---:|---:|
| 0 | (0) | 1 |
| 1 | (1) | 5 |

The updated claim is computed by the verifier via interpolation.
Again, this round is degree 1, so:

$$
\mathrm{claim}_2 = g_2(5)
= \mathrm{mle\_update}(g_2(0), g_2(1), 5)
= (7-3)\cdot 5 + 3
= 23 \equiv 6 \pmod{17}.
$$

Again, you do **NOT** need to implement the claim update.

---

# Round 3 (Eliminate $x_3$)

Finally we define

$$
g_3(t) = f(r_1,r_2,t).
$$

At this point only one Boolean variable remains. Current table from Round 2:

$$
[1,5].
$$

Since $x_3$ is now the least-significant bit in this 1-variable table:

| x3 | index | value |
|---|---|---|
|0|0|1|
|1|1|5|

So

$$
g_3(0)=1
$$

and

$$
g_3(1)=5.
$$

**The following is done by the verifier, NOT the prover, so you do NOT need to implement it !!!**

Consistency check:

$$
g_3(0)+g_3(1)=1+5=6 \equiv \text{claim}_2 \pmod{17},
$$

which matches $\text{claim}_2=6$ from Round 2.


The verifier privately samples a final random field element $r_3 = 11 \in \mathbb{F}_{17}$. The verifier then checks if $g_3(r_3) \equiv f(r_1, r_2, r_3)$: 

$$
g_3(r_3) = \mathrm{mle\_update}(g_3(0), g_3(1), r_3)
$$

$$
= (5 - 1) \cdot 11 + 1 = 45 \equiv 11 \pmod{17}
$$


$$
f(r_1, r_2, r_3) = r_1 + 2r_2 + 4r_3 = 8 + 2\cdot 5 + 4\cdot 11 = 62 \equiv 11 \pmod{17} 
$$

Since the two values are equivalent, the verifier **Accepts** the prover's effort. At any point in this process, if a claim is not satisfied, the verifier **Rejects**.


# Compositions of Multilinear Polynomials

In practice, the polynomial $f$ is often built from several multilinear polynomials. For example:

$$
X = [x_1, x_2, x_3, \ldots, x_n]
$$

$$
f(X) = a(X)b(X) + c(X)
$$

or

$$
f(X) = a(X)b(X)c(X).
$$

For these polynomials, it's not enough to just add up the evens and odds. When we did SumCheck for $f(X) = g(X)$, in each round, we computed a **linear** polynomial $g_i(t)$. A line only needs 2 unique points to characterize it, so we could just compute the sum of evens and sum of odds to get 2 points.

However, if we had $f(X) = a(X)b(X) + c(X)$, the degree of this polynomial is $2$, because we are multiplying two multi**linear** polynomials together ($a$ and $b$). So in round 1, we're computing

$$
f_1(t) = \sum_{(x_2 \ldots x_n)\in{\lbrace 0,1\rbrace}^{n-1}} a(t,x_2,\ldots,x_n)\cdot b(t,x_2,\ldots,x_n) + c(t,x_2,\ldots,x_n).
$$

$f_1(t)$ is actually a degree-$2$ polynomial, so we need $3$ points to characterize it. Having $f_1(0)$ (sum of even entries) and $f_1(1) (sum of odd entries) only gives us 2 of the needed points. We need to compute $f_1(2)$ as well.

To compute $f_1(0), f_1(1), f_1(2)$, we do the following

```
for each pair of even and odd indices:
    for each polynomial:
        set the even entry as the zero_val
        set the odd entry as the one_val
        use the MLE Update formula to get `MLE_Update(zero_val, one_val, 2)`
    
    # compute_composition() computes the formula, e.g. a*b + c
    f_pair_zero = compute_composition(a(zero_val), b(zero_val), c(zero_val))
    f_pair_one  = compute_composition(a(one_val), b(one_val), c(one_val))
    f_pair_two  = compute_composition(a(two_val), b(two_val), c(two_val))
    
    f_0 += f_pair_zero
    f_1 += f_pair_one
    f_2 += f_pair_two
```

This shows the case for a degree-$2$ polynomial. In general, if we have a degree-$d$ polynomial, we need to compute the round evaluations $f_i(0), f_i(1), \ldots, f_i(d)$ using the procedure above.
 
# Pseudocode example
Below is pseudocode describing the full protocol end-to-end

```
SumCheck(tables_list, composition_function, num_vars)

num_rounds = num_vars
degree = get_polynomial_degree(composition_function)

for round_idx in num_rounds:
   
    table_0 = tables_list[0]
    num_entries = table_length(table_0) # all tables should be the same length each round
    round_evals = []
    
    
    # for all code below, the order of operations is up to you as long as it is functionally correct. The following is just a general template
    
    for idx in num_entries/2:
        extensions = get_needed_extensions(tables_list, degree)
        composition_evals = compute_composition(composition_function, extensions)
        accumulated_composition_evals += composition_evals

    round_evals[round_idx] = accumulated_composition_evals

    # remember which round you need this in!
    claimed_sum = compute_claimed_sum()

    # remember which round you dont need to this for!
    round_challenge = get_random_challenge()
    mle_update(tables_list, round_challenge)

return claimed_sum, round_evals

```

# Intuition behind the MLE Update formula

**This section isn't required to implement the assignment but is provided to give additional context on what's happening.**

The MLE Update is the special case of the slope-intercept form where the two $x$-coordinates are $0$ and $1$.

Recall, given two points $(x_1, y_1)$ and $(x_2, y_2)$, the equation of the line  between the two points $y = f(x)$ is 

$$
y = mx + b = \frac{y_2 - y_1}{x_2 - x_1}x + b 
$$

We're usually applying MLE Update to a "zero-entry" and a "one-entry", e.g., two points $(0, f(0))$ and $(1, f(1))$. Then, the equation for $y$ becomes

$$
y = \frac{f(1)-f(0)}{1 - 0}x + f(0) \rightarrow y = (f(1) - f(0))x + f(0)
$$

This is precisely the equation we need to compute the target value $x$ given the two coordinates we had. Substituting them in as the following:

$$
z := f(0)
$$

$$
o := f(1)
$$

$$
t := x 
$$

We get

$$
\mathrm{mle\_update}(z,o,t) = (o - z)t + z.
$$

So in other words, the MLE Update formula is how we linearly extrapolate to the value $t$ given the two values. This property works because of the multilinear property of the polynomial $f$. 

Going back to the 3-variable example above, in round 1, if we looked at the values at indices 2 and 3 in the table, we have $z := f(1,0,0)$ and $o := f(1,0,1)$. 

We can think of these two points as a line $y_{1,0} = f(1,0, x_1) = f_{1,0}(x_1)$. Because $x_3$ and $x_2$ are fixed to $1$ and $0$, respectively, they fall out the equation. Then, we can use these two points to extrapolate to some point $f_{1,0}(t) = (o-z)t+z$. For this $3$-variable example, in Round 1, we're doing for every combination of $(x_3, x_2)$ to get the corresponding evaluation at the desired target value.

In general, for a $n$-variable polynomial in Round $i$, we'll be using the MLE Update formula over $f(x_n, x_{n-1}, \ldots, x_{i+1}, x_i, r_{i-1}, \ldots r_1)$, where $x_n, \ldots, x_{i+1}$ are binary values we're summing over, $x_i$ is the variable we're trying to solve for, and $r_{i-1} \ldots r_1$ are the challenges sampled during rounds $1$ to $i - 1$.


# Hint for MLE Update

MLE Update involves a modular multiplication to get the target value. Since MLE Update is a linear extrapolation, is there a smarter way to do this? 

Before you try to optimize for this, make sure the standard approach with modular multiplication is functionally correct!
