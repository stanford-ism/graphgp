# %% Imports
import json
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

with open("../output/simple_results_32.json") as f:
    results = json.load(f)["results"]

# %% Plot

k_values = sorted({r["k"] for r in results})
cmap_cuda = plt.get_cmap("Blues")
cmap_jax  = plt.get_cmap("Oranges")

fig, axes = plt.subplots(1, 3, figsize=(10,5), dpi=300)

for i, k in enumerate(k_values):
    for cuda, cmap in [(True, cmap_cuda), (False, cmap_jax)]:
        color = cmap((i + 1) / (len(k_values) + 1))
        runs = [r for r in results if r["k"] == k and r["cuda"] == cuda and r["status"] == "ok"]
        runs.sort(key=lambda r: r["n"])
        ns   = [r["n"] for r in runs]

        axes[0].plot(ns, [r["graph_build_time_s"] for r in runs], "o-", color=color)
        axes[1].plot(ns, [r["forward_mean_s"]     for r in runs], "o-", color=color)
        axes[2].plot(ns, [r["forward_memory_mb"] / 1024 for r in runs], "o-", color=color)

for ax in axes:
    ax.set(box_aspect=1, xlim=(1e6, 1e9), xticks=[1e6, 1e7, 1e8, 1e9], xscale="log", yscale="log", xlabel="Points")

axes[0].set(
    ylim=(1e-1, 3e2),
    # ylabel="Time [s]",
    title="Graph build time [s]",
)
axes[1].set(
    ylim=(1e-3, 3e0),
    # ylabel="Time [s]",
    title="Forward pass time [s]",
)
axes[2].set(
    ylim=(1e-1, 1e2),
    # ylabel="Memory [GB]",
    title="Forward pass memory [GB]",
)

gray_cmap = plt.get_cmap("Greys")
for i, k in enumerate(k_values):
    color = gray_cmap(0.35 + 0.45 * i / max(len(k_values) - 1, 1))
    axes[0].plot([], [], "o", color=color, label=f"k={k}")
axes[0].plot([], [], "-", color=cmap_jax(0.6),  label="JAX")
axes[0].plot([], [], "-", color=cmap_cuda(0.6), label="CUDA")
axes[0].legend(loc="lower right", fontsize=9)

plt.tight_layout()
# plt.savefig("../../Overleaf/graphgp-pai26/figures/benchmark.pdf", dpi=300)
plt.show()

# %%
