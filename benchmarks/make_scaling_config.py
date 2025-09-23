import json

config = {
  "defaults": {
    "covariance": {"matern_p": 0, "discrete_cov": True, "r_min": 1e-5, "r_max": 10.0, "n_bins": 1000},
    "distribution": {"type": "gaussian"},
    "graph": {"strict": True, "serial_depth": False, "fuse": True},
    "function": "forward",
    "timing_runs": 5,
    "cuda": True,
    "seed": 137
  },
  "runs": [],
}


d = 3

for n in [1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 300_000_000, 600_000_000, 1_000_000_000]:

    for k in [1, 2, 4, 8, 16]:
        if n * (d + d + k + 4) > 15 * 1e9:
            continue
        for cuda in [True]:
            run_config = {
                "n": n,
                "n0": 1000,
                "d": d,
                "k": k,
                "cuda": cuda,
            }
            config["runs"].append(run_config)

    config["runs"].append({
        "n": n,
        "n0": 1000,
        "d": d,
        "k": 1,
        "function": "fft",
    })

with open("benchmarks/cuda_scaling.json", "w") as f:
    json.dump(config, f, indent=2)