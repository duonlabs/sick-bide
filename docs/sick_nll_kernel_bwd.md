# How does the backward pass of the fused NLL loss and softmax kernel work?

The backward pass of the fused NLL loss and softmax kernel is a crucial part of the training process. It involves computing the gradients of the loss with respect to the observations and the model parameters.

# Problem with the naïve implementation
You can find the vanilla pytorch implementation of BIDE in `sick_bide/reference.py`. To compute the nll forward and backward passes, we need to materialize the $2^n$ binary possibilities. This consumes unecessarily a lot of memory or is simply is not feasible for large $n$.

# Notation

* $\mathcal{L}$: Loss function 
* $n$, *N*: index of the minibatch example and number of examples in the minibatch.
* $k$ and $K$: index of the binary input value and number of binary input combinations.
* $b_{kj}$: representation of bit $j$ (-1 if the bit is 0 else 1) for the $k$-th binary input value
* $W$: weights of the first layer of the BIDE MLP
* $r$: weights of the second layer of the BIDE MLP
* $h_{ki}$: value of the $i$-th hidden unit for the $k$-th binary input value
* $l_{k}$: logit of the $k$-th binary input value
* $p_{k}$: probability of the $k$-th binary input value
* $y_n$: index of the target value for the $n$-th example
* $\delta_{ky_n}$: kronecker delta, equal to 1 if $k = y_n$ and 0 otherwise
* $m$: maximum value of the logits (used for softmax numerical stability)
* $s$: sum of the exponentials of the logits (used for softmax normalization)

# Calculations
We want to compute $\frac{\partial \mathcal{L}}{\partial r}$ and $\frac{\partial \mathcal{L}}{\partial W}$ in an efficient way. Here are the calculations assuming mean reduction for the loss over the minibatch:
## Gradient of *r*
```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial r_i} &= \frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} \frac{\partial l_k}{\partial r_i} \frac{\partial \mathcal{L}}{\partial l_k} \\
&= \frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} h_{ki} (p_k - \delta_{ky_n}) \\
&= \frac{1}{N} \sum_{n=1}^{N} \left( \sum_{k=1}^{K} h_{ki} p_k - \sum_{k=1}^{K} h_{ki} \delta_{ky_n} \right) \\
&= \sum_{k=1}^{K} h_{ki} p_k - \frac{1}{N} \sum_{n=1}^{N} h_{y_ni} \\
&= \sum_{k=1}^{K} h_{ki} \frac{\exp(l_k - m)}{s} - \frac{1}{N} \sum_{n=1}^{N} h_{y_ni} \\
&= \frac{1}{s} \sum_{k=1}^{K} h_{ki} \exp(l_k - m) - \frac{1}{N} \sum_{n=1}^{N} h_{y_ni} \\
\end{align*}
```
In pratice, $K \gg N$, and any $O(KN)$ operation is not feasible.
Fortunately, we can split the sum in two with one iteration over $K$ and one iteration over $N$.
Note that saving all the *h_ki* in memory is not an option, so we either have to compute the $K$ sum in forward pass or leverage gradient checkpointing.

## Gradient of *W*
```math
\begin{align*}
\frac{\partial \mathcal{L}}{\partial W_{ij}} &= \frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} \frac{\partial h_ki}{\partial W_{ij}} \frac{\partial l_k}{\partial h_{ki}} \frac{\partial \mathcal{L}}{\partial l_k} \\
&= \frac{1}{N} \sum_{n=1}^{N} \sum_{k=1}^{K} b_{kj} \mathbb{1}_{h_{ki} \gt 0} r_i (p_k - \delta_{ky_n}) \\
&= \frac{1}{N} \sum_{n=1}^{N} \left( \sum_{k=1}^{K} b_{kj} \mathbb{1}_{h_{ki} \gt 0} r_i p_k - \sum_{k=1}^{K} b_{kj} \mathbb{1}_{h_{ki} \gt 0} r_i \delta_{ky_n} \right) \\
&= \sum_{k=1}^{K} b_{kj} \mathbb{1}_{h_{ki} \gt 0} r_i p_k - \frac{1}{N} \sum_{n=1}^{N} b_{y_nj} \mathbb{1}_{h_{y_ni} \gt 0} r_i \\
&= \sum_{k=1}^{K} b_{kj} \mathbb{1}_{h_{ki} \gt 0} r_i \frac{\exp(l_k - m)}{s} - \frac{1}{N} \sum_{n=1}^{N} b_{y_nj} \mathbb{1}_{h_{y_ni} \gt 0} r_i \\
&= \frac{1}{s} \sum_{k=1}^{K} b_{kj} \mathbb{1}_{h_{ki} \gt 0} r_i \exp(l_k - m) - \frac{1}{N} \sum_{n=1}^{N} b_{y_nj} \mathbb{1}_{h_{y_ni} \gt 0} r_i \\
\end{align*}
```
In the same way, we can split the sum in two with one iteration over $K$ and one iteration over $N$.

# Efficient implementation
The computation of the forward pass, the gradient of $r$ and the gradient of $W$ can all be broken down into two loops: one over the minibatch examples and one over the binary input combinations. This means that if we can implement an efficient kernel for a program running over the binary input combinations, we can compute efficiently everything we need to train the BIDE model.