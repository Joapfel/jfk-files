# JFK-Files-Part-1_page_15.png

Convert to Markdown② Indicate that $S_0$ is an initial sum, and $S_n$ is a sequence. Using the definition of $S_{n+1}$, prove by mathematical induction that for all positive integers $n$, $S_n$ is equal to the sum of the first $n$ terms of $\{S_k\}$.

$$ \text{The sequence } \{S_n\} \text{ defined as } S_0 = 2 \text{ and } S_{n+1} = S_n + 2 \text{ for } n \ge 1 $$

is convergent. Let's use the ratio test:

Let $r = \frac{S_{n+1}}{S_n}$. In this case,

$r = \frac{S_2}{S_1} = \frac{2+2}{2} = 1+1 = 2$

$a_1 = 2$

$$ \lim_{n \to \infty} r^n = \lim_{n \to \infty} 2^n = 2 \times 2 \times ... = 2^{\infty} $$

Since $r > 1$, the limit becomes infinite, which contradicts the condition for convergence. This means that the series diverges.
