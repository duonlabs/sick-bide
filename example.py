import math
import torch
import time
import numpy as np
import lovely_tensors as lt
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Tuple
from sick_bide.kernels.utils import value_to_binary_representation
from sick_bide.utils import EL_SIZE2UINT_DTYPE, NBITS2FLOAT_DTYPE
from sick_bide import BIDE

lt.monkey_patch()
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_float32_matmul_precision("medium")

N_STEPS = 2000
BATCH_SIZE = 64
MIN_LR = 1e-5
MAX_LR = 1e-1
N_BITS = 16


class MultipleBIDE(torch.nn.Module):
    def __init__(self, n_dists: int, n_bits: int = 16, hidden_factor: int = 2):
        super().__init__()
        self.n_bits = n_bits
        self.bide_hidden_size = n_bits * hidden_factor
        self.Ws = torch.nn.Parameter(torch.randn(n_dists, self.bide_hidden_size, self.n_bits))
        self.Ws.data.normal_(0, math.sqrt(2 / self.n_bits))
        self.rs = torch.nn.Parameter(torch.randn(n_dists, self.bide_hidden_size))
        self.rs.data.normal_(0, 1 / math.sqrt(self.bide_hidden_size))

    @torch.compile
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            x: [B]
        """
        W = self.Ws[x] # [B, H, N]
        r = self.rs[x] # [B, H]
        return W, r

true_dists = [
    torch.distributions.Normal(3, 1),
    torch.distributions.Gamma(2, 1),
    torch.distributions.MixtureSameFamily(
        torch.distributions.Categorical(torch.tensor([0.2, 0.8])),
        torch.distributions.Normal(torch.tensor([120.0, 132.0]), torch.tensor([1.0, 1.5]))
    ),
    torch.distributions.Pareto(1.0, 1.0),
]

def sample_true_dists(dist_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    samples = torch.empty(len(dist_ids), dtype=torch.float16, device=device)
    for i, dist in enumerate(true_dists):
        mask_i = dist_ids == i
        n_samples_i = (mask_i).sum()
        if n_samples_i == 0:
            continue
        samples_i = dist.sample((n_samples_i,)).to(device)
        samples[mask_i] = samples_i.to(samples.dtype)

    return samples

model = MultipleBIDE(len(true_dists), n_bits=N_BITS, hidden_factor=2).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR, betas=(0.9, 0.96))
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer,
    gamma=(MIN_LR / MAX_LR) ** (1 / N_STEPS)
)

# Lists to store frames for the animation
cdf_frames = []
density_frames = []
step_numbers = []
loss_h = []
loss_h_ema = []

X_test = torch.arange(len(true_dists), device=device)
xs = []
b = torch.linspace(-5, 5, 64, device=device, dtype=torch.float32)
for dist in true_dists:
    xs.append(
        (dist.mean if  torch.isfinite(dist.mean) else 0)
        + b * (dist.stddev if torch.isfinite(dist.stddev) else 5)
    )
xs = torch.stack(xs, dim=0) # [4, 32]
x_vals = ((xs[..., :-1] + xs[..., 1:]) / 2) # [4, 31]

for i in range(N_STEPS):
    if i == 1:
        t_start = time.time() # Skip the warmup step in the timing
    X = torch.randint(len(true_dists), (BATCH_SIZE,), device=device) # [B]
    y = sample_true_dists(X) # [B]
    W, r = model(X) # [B, H, N], [B, H]
    bide = BIDE(W, r)
    nll = bide.nll(y)# []
    if i > 25:
        loss_h.append(nll.item())
        if not loss_h_ema:
            loss_h_ema.append(loss_h[0])
        else:
            loss_h_ema.append(0.9 * loss_h_ema[-1] + 0.1 * nll.item())
    optimizer.zero_grad()
    nll.backward()
    optimizer.step()
    scheduler.step()
    print(f"Step {i}, NLL: {nll.item()} (ema: {loss_h_ema[-1] if loss_h_ema else nll.item()})")

    if i % 25 == 0:
        model.eval()
        with torch.no_grad():
            W, r = model(X_test) # [B, H, N], [B, H]
            bide = BIDE(W, r)
            log_cdf = bide.log_cdf(xs.to(torch.float16))
        # Save the data for this frame
        cdf_frames.append(log_cdf.exp().cpu().numpy())
        # Compute density
        p = torch.diff(log_cdf.exp(), dim=-1)
        d = (p / torch.diff(xs, dim=-1)).cpu().numpy() # [B, 100]
        density_frames.append(d)
        step_numbers.append(i)
print(f"Training took {time.time() - t_start:.2f} seconds")
print(f"Peak memory usage: {torch.cuda.max_memory_allocated() / 2**20:.2f} MB")
# Determine fixed y-axis limits
xs, x_vals = xs.cpu(), x_vals.cpu()
true_cdfs = []
true_densities = []
for i, true_dist in enumerate(true_dists):
    true_cdfs.append(true_dist.cdf(xs[i]))
    true_densities.append(true_dist.log_prob(x_vals[i]).exp())
true_cdfs = torch.stack(true_cdfs, axis=0) # [4, 32]
true_densities = torch.stack(true_densities, axis=0) # [4, 31]
max_cdf_y = 1.05  # # Max window size for CDF plots
max_density_y = 1.2 * true_densities.nan_to_num(-float("inf")).max(-1).values  # Max window size for density plots
# Create the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=len(X_test), figsize=(len(X_test)*6, 10))
if len(X_test) == 1:
    axes = axes[None, :]
# fig.tight_layout()

# Convert the frame lists to arrays for easier indexing
cdf_array = np.stack(cdf_frames, axis=0)   # [num_frames, B, 101]
density_array = np.stack(density_frames, axis=0) # [num_frames, B, 100]

lines_cdf = []
lines_density = []
true_lines_cdf = []
true_lines_density = []

# Initialize plots
for i in range(len(X_test)):
    # CDF
    line_cdf, = axes[0, i].plot(xs[i], cdf_array[0, i], label='Fitted CDF')
    true_line_cdf, = axes[0, i].plot(xs[i], true_cdfs[i], '--', label='True CDF', zorder=-1)
    axes[0, i].fill_between(xs[i], 0, cdf_array[0, i], alpha=0.2, color=line_cdf.get_color())
    axes[0, i].set_ylim(0, max_cdf_y)
    axes[0, i].set_xlabel('x')
    axes[0, i].set_ylabel('CDF')
    axes[0, i].legend()
    lines_cdf.append(line_cdf)
    true_lines_cdf.append(true_line_cdf)

    # Density
    line_density, = axes[1, i].plot(x_vals[i], density_array[0, i], label='Fitted Density')
    true_line_density, = axes[1, i].plot(x_vals[i], true_densities[i], '--', label='True Density', zorder=-1)
    axes[1, i].fill_between(x_vals[i], 0, density_array[0, i], alpha=0.2, color=line_density.get_color())
    axes[1, i].set_ylim(0, max_density_y[i] if np.isfinite(max_density_y[i]) else 1.0)
    axes[1, i].set_xlabel('x')
    axes[1, i].set_ylabel('Density')
    axes[1, i].legend()
    lines_density.append(line_density)
    true_lines_density.append(true_line_density)

def init():
    # Global title
    fig.suptitle("Distributions estimations (CDF and Density)")
    for i in range(len(X_test)):
        lines_cdf[i].set_data(xs[i], cdf_array[0, i])
        lines_density[i].set_data(x_vals[i], density_array[0, i])
    return lines_cdf + lines_density

def animate(frame_idx):
    for i in range(len(X_test)):
        lines_cdf[i].set_data(xs[i], cdf_array[frame_idx, i])
        lines_density[i].set_data(x_vals[i], density_array[frame_idx, i])
        # Clear previous fill_between collections
        for collection in axes[0, i].collections:
            collection.remove()
        for collection in axes[1, i].collections:
            collection.remove()
        
        axes[0, i].fill_between(xs[i], 0, cdf_array[frame_idx, i], alpha=0.2, color=lines_cdf[i].get_color())
        axes[1, i].fill_between(x_vals[i], 0, density_array[frame_idx, i], alpha=0.2, color=lines_density[i].get_color())
    return lines_cdf + lines_density

ani = animation.FuncAnimation(fig, animate, frames=len(cdf_frames), init_func=init, interval=100, blit=True, repeat=True)
# Save the animation
ani.save("training_animation.gif", writer="pillow", fps=10)
# Save the last frame
plt.savefig("training_plot.png")
# Save the loss plot
plt.clf()
plt.plot(loss_h, label="NLL")
plt.plot(loss_h_ema, label="NLL EMA")
plt.xlabel("Step")
plt.ylabel("NLL")
plt.legend()
plt.savefig("training_loss.png")


