import json

config = {
    "defaults": {
        "covariance": {
            "matern_p": 0,
            "variance": 1.0,
            "cutoff": 1.0,
            "r_min": 1e-5,
            "r_max": 10.0,
            "n_bins": 1000,
        },
        "distribution": {"type": "gaussian"},
        "graph": {"strict": True, "serial_depth": False, "fuse": True},
        "function": "forward",
        "timing_runs": 5,
        "seed": 137,
    },
    "runs": [],
}

d = 3
k = 30

for n in [10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000]:
    for cuda in [True, False]:
        config["runs"].append({
            "n": n,
            "n0": 1000,
            "d": d,
            "k": k,
            "cuda": cuda,
            "function": "forward",
        })

with open("gpgpu_benchmark/graphgp_config.json", "w") as f:
    json.dump(config, f, indent=2)
