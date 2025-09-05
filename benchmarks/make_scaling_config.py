import json

config = {
  "defaults": {
    "covariance": {"matern_p": 0, "discrete_cov": True, "r_min": 1e-5, "r_max": 10.0, "n_bins": 1000},
    "distribution": {"type": "gaussian"},
    "graph": {"strict": True, "factor": 5.0},
    "warmup_runs": 1,
    "timing_runs": 5,
    "cuda": False,
    "seed": 137
  },
  "runs": [],
}

for n in [1_000_000_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000, 300_000_000]:
    for k in [4, 8, 16]:
        for strict in [True]:
            for cuda in [True]:
                run_config = {
                    "n": n,
                    "n0": 10000,
                    "d": 3,
                    "k": k,
                    "cuda": cuda,
                }
                if not strict:
                    run_config["graph"] = {"strict": False, "factor": config["defaults"]["graph"]["factor"]}
                config["runs"].append(run_config)

with open("benchmarks/cuda_scaling_config.json", "w") as f:
    json.dump(config, f, indent=2)