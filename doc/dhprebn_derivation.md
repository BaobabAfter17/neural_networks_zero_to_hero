# Deriving `dhprebn`: Backpropagating Through BatchNorm in One Shot

This note gives the core derivation for makemore part4 exercise 2 — backpropagating the gradient from `dhpreact` all the way to `dhprebn` (through the entire BatchNorm) in one shot, rather than node-by-node.

## 1. Forward pass (per-column, independent)

BatchNorm normalizes each column independently along the batch dimension (dim 0, size $n$). Fixing one column, let its $n$ inputs be $x_i = $ `hprebn[i]`:

$$\mu = \frac1n\sum_k x_k,\qquad \sigma^2 = \frac{1}{n-1}\sum_k (x_k-\mu)^2$$

$$\hat\sigma = (\sigma^2+\epsilon)^{-1/2}=\mathtt{bnvar\_inv}$$

$$\texttt{bnraw}_i = (x_i-\mu)\,\hat\sigma,\qquad \texttt{hpreact}_i = \gamma\,\texttt{bnraw}_i+\beta$$

where $\gamma=$ `bngain` and $\beta=$ `bnbias`.

Given the upstream gradient `dhpreact`, first obtain the gradient with respect to `bnraw`:

$$g_i \equiv \frac{\partial L}{\partial\,\texttt{bnraw}_i}=\gamma\cdot\texttt{dhpreact}_i$$

The goal is

$$\frac{\partial L}{\partial x_i}=\sum_j g_j\,\frac{\partial\,\texttt{bnraw}_j}{\partial x_i}$$

Key point: $x_i$ affects $\mu$, $\hat\sigma$, and $\texttt{bnraw}_j$ simultaneously, so none of them can be treated as constant.

## 2. Three local derivatives

**(1) With respect to $(x_j-\mu)$:**

$$\frac{\partial (x_j-\mu)}{\partial x_i}=\delta_{ij}-\frac1n$$

**(2) With respect to $\hat\sigma$:** first compute the derivative of $\sigma^2$ with respect to $x_i$. Using $\sum_k(x_k-\mu)=0$:

$$\frac{\partial\sigma^2}{\partial x_i}=\frac{1}{n-1}\sum_k 2(x_k-\mu)\Big(\delta_{ik}-\frac1n\Big)=\frac{2}{n-1}(x_i-\mu)$$

and since $\hat\sigma=(\sigma^2+\epsilon)^{-1/2}\Rightarrow \dfrac{\partial\hat\sigma}{\partial\sigma^2}=-\tfrac12\hat\sigma^3$, we get

$$\frac{\partial\hat\sigma}{\partial x_i}=-\frac12\hat\sigma^3\cdot\frac{2}{n-1}(x_i-\mu)=-\frac{\hat\sigma^3(x_i-\mu)}{n-1}$$

**(3) Combining into $\dfrac{\partial\,\texttt{bnraw}_j}{\partial x_i}$**, where $\texttt{bnraw}_j=(x_j-\mu)\hat\sigma$:

$$\frac{\partial\,\texttt{bnraw}_j}{\partial x_i}=\Big(\delta_{ij}-\frac1n\Big)\hat\sigma+(x_j-\mu)\Big(-\frac{\hat\sigma^3(x_i-\mu)}{n-1}\Big)$$

Substituting back $\texttt{bnraw}=(x-\mu)\hat\sigma$, so that $(x_j-\mu)(x_i-\mu)\hat\sigma^3=\hat\sigma\,\texttt{bnraw}_i\texttt{bnraw}_j$:

$$\boxed{\;\frac{\partial\,\texttt{bnraw}_j}{\partial x_i}=\hat\sigma\Big(\delta_{ij}-\frac1n\Big)-\frac{\hat\sigma}{n-1}\,\texttt{bnraw}_i\,\texttt{bnraw}_j\;}$$

## 3. Summing for the backward pass

$$\frac{\partial L}{\partial x_i}=\sum_j g_j\,\frac{\partial\,\texttt{bnraw}_j}{\partial x_i}
=\hat\sigma\Big[g_i-\frac1n\sum_j g_j-\frac{\texttt{bnraw}_i}{n-1}\sum_j g_j\,\texttt{bnraw}_j\Big]$$

The three terms correspond to: the **passthrough term**, the **mean-subtraction gradient**, and the **variance-subtraction gradient**.

Substituting back $g=\gamma\cdot\texttt{dhpreact}$ and factoring out $\gamma$, in vector form (`sum(0)` denotes summation along the batch dimension with broadcasting):

$$\texttt{dhprebn}=\frac{\gamma\cdot\mathtt{bnvar\_inv}}{n}\Big(n\cdot\texttt{dhpreact}-\texttt{dhpreact.sum(0)}-\frac{n}{n-1}\,\texttt{bnraw}\odot(\texttt{dhpreact}\odot\texttt{bnraw}).\texttt{sum(0)}\Big)$$

Corresponding code:

```python
dhprebn = bngain*bnvar_inv/n * (n*dhpreact - dhpreact.sum(0)
          - n/(n-1)*bnraw*(dhpreact*bnraw).sum(0))
```

## 4. Two remarks

- **The `n/(n-1)` factor comes from Bessel's correction**: the forward `bnvar` divides by $n-1$ rather than $n$. This is also why this step can only achieve `approximate: True` (maxdiff ~9e-10) rather than `exact: True` — numerically it differs slightly from the node-by-node backprop.
- The second and third terms are exactly the **coupling** among samples within a batch: because $\mu,\sigma$ depend on the entire batch, the gradient of each $x_i$ is "pulled back" partly by the batch's mean/variance gradients. This is the mathematical root of why BatchNorm makes samples within the same batch influence one another.
