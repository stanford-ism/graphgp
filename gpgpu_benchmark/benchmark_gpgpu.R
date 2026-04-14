#!/usr/bin/env Rscript
#
# Benchmark GpGpU forward pass (likelihood evaluation) for comparison with graphgp.
#
# Usage:
#   Rscript gpgpu_benchmark/benchmark_gpgpu.R [output_file]
#
# Requires: GpGpU, FNN, jsonlite

library(GpGpU)
library(jsonlite)

# R wrapper has a bug (missing covfun_name arg), so call C directly
gpgpu_loglik <- function(covparms, covfun_name, y, X, locs, NNarray) {
  .Call("_GpGpU_vecchia_profbeta_loglik_gpu", PACKAGE = "GpGpU",
        covparms, covfun_name, y, X, locs, NNarray)
}

# --- Configuration ---
n_values <- c(10000, 30000, 100000, 300000, 1000000, 3000000, 10000000)
k <- 30
d <- 3
timing_runs <- 5
seed <- 137

# Exponential covariance: covparms = c(variance, range, nugget)
# Matches graphgp matern_kernel(p=0, variance=1.0, cutoff=1.0)
covparms <- c(1.0, 1.0, 1e-5)
covfun_name <- "exponential_isotropic"

# Output file
args <- commandArgs(trailingOnly = TRUE)
output_file <- if (length(args) > 0) args[1] else "gpgpu_benchmark/gpgpu_results.json"

results <- list()

for (n in n_values) {
  cat(sprintf("n = %d\n", n))
  set.seed(seed)

  # Generate standard normal points in R^d
  locs <- matrix(rnorm(n * d), ncol = d)
  y <- rnorm(n)
  X <- matrix(1, n, 1)

  # Random ordering (fast, avoids O(n^2) maxmin)
  t_order_start <- proc.time()["elapsed"]
  ord <- sample(1:n)
  locs <- locs[ord, ]
  y <- y[ord]
  X <- X[ord, , drop = FALSE]
  t_order <- proc.time()["elapsed"] - t_order_start

  # Find nearest neighbors
  gc()
  t_nn_start <- proc.time()["elapsed"]
  NNarray <- find_ordered_nn(locs, m = k)
  t_nn <- proc.time()["elapsed"] - t_nn_start

  cat(sprintf("  neighbors: %.3f s\n", t_nn))

  # Warmup (first call may have overhead)
  gc()
  dummy <- gpgpu_loglik(covparms, covfun_name, y, X, locs, NNarray)

  # Timed runs
  times <- numeric(timing_runs)
  for (i in seq_len(timing_runs)) {
    gc()
    t_start <- proc.time()["elapsed"]
    result <- gpgpu_loglik(covparms, covfun_name, y, X, locs, NNarray)
    times[i] <- proc.time()["elapsed"] - t_start
    cat(sprintf("  loglik run %d: %.4f s\n", i, times[i]))
  }

  mean_time <- mean(times)
  std_time <- sd(times)
  cat(sprintf("  loglik mean: %.4f +/- %.4f s\n\n", mean_time, std_time))

  results[[length(results) + 1]] <- list(
    parameters = list(n = n, k = k, d = d, cuda = TRUE),
    graph_timings = list(ordering = t_order, find_ordered_nn = t_nn),
    timing = list(mean = mean_time, std = std_time, times = times)
  )
}

# Save results
output <- list(
  timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
  config = list(
    k = k, d = d,
    covfun_name = covfun_name,
    covparms = covparms,
    timing_runs = timing_runs,
    seed = seed,
    ordering = "random"
  ),
  results = results
)

write_json(output, output_file, auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("Results saved to %s\n", output_file))
