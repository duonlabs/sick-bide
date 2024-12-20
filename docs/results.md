# Bide viability analysis

# Optained throughput
We ran experiments on the use case described in [theoretical ceiling](#theoretical-ceiling) (LLM setting, batch_size=64, context_size=512, 16bites-32hidden_dim-BIDE, float16 computations).

On an [NVIDIA GeForce RTX 4060 Laptop GPU](https://www.techpowerup.com/gpu-specs/geforce-rtx-4060-mobile.c3946):
* 11.61 TFLOPS compute
* 256.0 GB/s bandwidth

We obtain:
Naive implementation:
* integral kernel:
    - Kernel duration: $138.33ms$
    - Throughput: $7.23 run/s$
    - Effective FLOPs: $7.23*1.21\thickapprox8.72 TFLOPs$ (75.1% of the GPU theoretical peak)
    - Effective Memory bandwidth: $7.23*17.89\thickapprox129.34 MB/s$ (0.05% of the GPU theoretical peak)

Pre-computing implementation:
* integral kernel:
    - Kernel duration: $63.18ms$
    - Throughput: $15.83 run/s$
    - Effective FLOPs: $15.83*0.150\thickapprox2.38TFLOPs$ (20.49% of the GPU theoretical peak)
    - Effective Memory bandwidth: $15.83*0.537\thickapprox8.51 GB/s$ (3.33% of the GPU theoretical peak)