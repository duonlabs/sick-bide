# Bide viability analysis

# Theoretical ceiling

## Notations

Numbers:
* $B$: Number of bits of BIDE
* $H$: Hidden size of BIDE
* $F$: Element size of BIDE dtype
* $\mathcal{B}$: Batch size
* $\mathcal{K}$: Input dimension ($2^B$)
* $N$: Batch dimension block size
* $K$: Input dimension block size

Tensors:
* $W$: Weights of BIDE 1st layer, shape $(N, H, B)$
* $r$: Weights of BIDE 2nd layer, shape $(N, H)$
* $b$: input values represented as bits, shape $(N, K, B)$
* $z$: intermediate values of BIDE, shape $(N, K, H)$
* $h$: activations of BIDE's hidden layer, shape $(N, K, H)$
* $l$: logits (output) of BIDE, shape $(N, K)$
* $m$: max logit of BIDE, shape $(N)$
* $s$: sum of the exponential of the logits (normalization constant), shape $(N)$
* $p$: probabilities of BIDE, shape $(N, K)$

# Motivation

We would like to estimate the throughput needed to viably run BIDE on a device. One particular use case that interests us is the use of BIDE as a LLM head.\
In particular let's consider as a baseline the [nano-gpt repository](https://github.com/karpathy/nanoGPT) which is a minimalistic implementation of GPT-2.\
Training this type of model on modern GPUs with mixed precision, we can expect roughly that the final layer + loss computation should be in the order of $10ms$ for a batch size of $32$ with a context size of $512$.\
If we want to use BIDE as a LLM head (either to serve as an implicit distribution over tokens or over a real number) we would like a reasonable precision like 16 bits.\
For this setting to be viable we can ask the simple question:\
How many memory bandwidth / FLOPs are required to run 16bits-BIDE with $\mathcal{B}=32*512=16384$ such that the loss computation in float16 is ~$10ms$?

## On-Device input materialization

Let's first consider a simple algorithm were we burteforce compute by generating the inputs of BIDE on the device using bitwise operations.\
This way we only need to load the weights and performs gemm and reduce operations.\

### Compute m and s (integral kernel)

Memory:
* Load $W$: $FNHB$
* Load $r$: $FNH$
* Write $m$: $FN$
* Write $s$: $FN$

Total: $FN(H(B+1)+2)$

Compute:
* Compute $b$ (>>, &, fma): $KB3$
* Compute $z$ (gemm): $NKHB2$
* Compute $h$ (relu): $NKH$
* Compute $l$ (gemm): $NKH2$
* Compute $m$ (max): $NK$
* Compute $s$ (-m, exp, sum): $NK3$

Total: $K(3B+N(H(B2+3)+4))$

Now let's plugin the numbers for our baseline:
* $B=16$
* $H=32$ (2x the input size, default setting)
* $\mathcal{B}=16384$
* $\mathcal{K}=2^B=65536$
* $F=2$

Which gives:
* Memory:
    * Load $W$: $2*16384*32*16 = 16 777 216 = 16.78Mb$
    * Load $r$: $2*16384*32 = 1 048 576 = 1.05Mb$
    * Write $m$: $2*16384 = 32 768 = 32.77kb$
    * Write $s$: $2*16384 = 32 768 = 32.77kb$
    * Total: $17 891 328 = 17.89Mb$
* Compute:
    * Compute $b$: $65536*16*3 = 3 145 728 = 3.15Mops$
    * Compute $z$: $16384*65536*32*16*2 = 1 099 511 627 776 = 1.1Tops$
    * Compute $h$: $16384*65536*32 = 34 359 738 368 = 34.36Gops$
    * Compute $l$: $16384*65536*32*2 = 68 719 476 736 = 68.72Gops$
    * Compute $m$: $16384*65536 = 1 073 741 824 = 1.07Gops$
    * Compute $s$: $16384*65536*3 = 3 221 225 472 = 3.22Gops$
    * Total: $1 206 888 955 904 = 1.21Tops$

As we aim for $10ms$, this means we need a GPU with at least $120.69TFLOPs$ and $1.79GB/s$ of memory bandwidth. The memory bandwidth is negligible by today's standards, but the FLOPs are considerable.\
We probably need some ways to reduce the FLOPs needed to run BIDE.

## Pre-computing blocks
We need to find a way to reduce the number of FLOPs needed to run BIDE compared to the naive implementation.
The naive implementation is heavily dominated by the gemm operation between $b$ and $W$.
However there is a lot of structure in the computation since $b$ is just an enumeration of all binary combinations of $-1$ and $1$. 
Indeed, if we compute the intermediate values $\tilde{z}_1$ and $\tilde{z}_2$ corresponding to $z$ computed partially over respectively the first $\frac{N}{2}$ input bits and last $\frac{N}{2}$ input bits, then any $z$ can be recovered by a sum of the right $\tilde{z}_1$ and $\tilde{z}_2$.\
Let's reconduct the analysis with this new structure.

### Compute m and s (integral kernel)

Let's introduce new notation:
* $\tilde{K}=2^{\frac{\log_2{K}}{2}}=\sqrt{2^{log_2{K}}}=\sqrt{K}$: the number of combinations of inputs for half the bits of BIDE.
* $\tilde{b}_1$ and $\tilde{b}_2$: Respectively the first and last half of the input bits representation
* $\tilde{z}_1$ and $\tilde{z}_2$: The intermediate values of BIDE computed over respectively $\tilde{b}_1$ and $\tilde{b}_2$

Memory:
* Load $\tilde{z}_1$: $FNH\tilde{K}$
* Load $\tilde{z}_2$: $FNH\tilde{K}$
* Load $r$: $FNH$
* Write $m$: $FN$
* Write $s$: $FN$

Total: $FN(H(2^{B/2+1}+1)+2)$

Compute:
* Compute $b1$ (>>, &, fma): $\tilde{K}B3$
* Compute $b2$ (>>, &, fma): $\tilde{K}B3$
* Compute $\tilde{z}_1$ (gemm): $N\tilde{K}HB2$
* Compute $\tilde{z}_2$ (gemm): $N\tilde{K}HB2$
* Compute $z$ (+): $NKH$
* Compute $h$ (relu): $NKH$
* Compute $l$ (gemm): $NKH2$
* Compute $m$ (max): $NK$
* Compute $s$ (-m, exp, sum): $NK3$

Total: $6\tilde{K}B+N(4\tilde{K}HB+K(4H+4))$

Now if we plugin the numbers for our baseline (same as precedent section plus $\tilde{K}=256$), we obtain:
* Memory:
    - Load $\tilde{z}_1$: $2*16384*32*256 = 268 435 456 = 268.44Mb$
    - Load $\tilde{z}_2$: $2*16384*32*256 = 268 435 456 = 268.44Mb$
    - Load $r$: $2*16384*32 = 1 048 576 = 1.05Mb$
    - Write $m$: $2*16384 = 32 768 = 32.77kb$
    - Write $s$: $2*16384 = 32 768 = 32.77kb$
    - Total: $537 985 024 = 537.99Mb$
* Compute:
    - Compute $b1$: $256*16*3 = 12 288 = 12.29kops$
    - Compute $b2$: $256*16*3 = 12 288 = 12.29kops$
    - Compute $\tilde{z}_1$: $16384*256*32*16*2 = 4 294 967 296 = 4.29Gops$
    - Compute $\tilde{z}_2$: $16384*256*32*16*2 = 4 294 967 296 = 4.29Gops$
    - Compute $z$: $16384*65536*32 = 34 359 738 368 = 34.36Gops$
    - Compute $h$: $16384*65536*32 = 34 359 738 368 = 34.36Gops$
    - Compute $l$: $16384*65536*32*2 = 68 719 476 736 = 68.72Gops$
    - Compute $m$: $16384*65536 = 1 073 741 824 = 1.07Gops$
    - Compute $s$: $16384*65536*3 = 3 221 225 472 = 3.22Gops$
    - Total: $150 323 879 936 = 150.32Gops$

this makes the GPU requires $53.8GB/s$ of memory bandwidth and $15.03TFLOPs$ of compute power which starts to be more reasonable.\
