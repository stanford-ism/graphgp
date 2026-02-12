---
title: 'GraphGP: Scalable Gaussian Processes with the Vecchia Approximation'
tags:
 - Python
 - astronomy
 - Gaussian process
authors:
 - name: Benjamin Dodge
   orcid: 0000-0002-2519-2219
   affiliation: 1
 - name: Philipp Frank
   affiliation: 1
 - name: Susan Clark
   affiliation: 1
affiliations:
 - name: Kavli Institute for Particle Astrophysics and Astronomy, United States
date: 2 February 2026
bibliography: paper.bib
---

# Summary


# Statement of need


# Vecchia approximation

Generating a Gaussian process realization at a finite collection of points involves sampling from a multivariate normal distribution. This can be accomplished by forming a covariance matrix $K$, computing the Cholesky factorization $K=L L^T$, and then applying this matrix to a vector of unit normal noise $v=L\xi$. Due to the lower-triangular nature of $L$, this is equivalent to sequential conditional generation, where each new value $v$ is conditioned on all previously generated values using the Gaussian conditioning formulas. This approach does not scale well, as it requires $O(N^2)$ memory and $O(N^3)$ time for the Cholesky factorization.

The Vecchia approximation conditions each new value only on $k$ previously generated values, where $k\ll N$. If the $k$ other values are well-chosen they can render the new value nearly conditionally independent of previous values. For stationary covariance kernels which decay with distance, choosing the $k$ nearest neighbors is a natural choice. This type of approximation yields a dense covariance matrix with long-range correlation by approximating the Cholesky factor of the precision matrix (inverse covariance) with a sparse matrix.

# Point ordering

Order matters in the Vecchia approximation. The impact of various heuristic ordering schemes on the approximation quality has been discussed in the literature.

We require an order that satisfies the following properties:
 - efficient *preceding* nearest neighbor search
 - 


# State of the field

The `NIFTy` software package includes tools for Bayesian inference of fields using Gaussian process priors. Particularly relevant is Iterative Charted Refinement, a Vecchia-like approximate method for generating Gaussian process realizations on irregular grids. GraphGP removes the need for grids, finding neighbors automatically with a k-d tree. It constructs valid parallel batches by analyzing the neighbor graph rather than manually truncating. It provides and exact inverse and determinant. It implements the key conditioning operation with a custom CUDA kernel. `NIFTy` is an inference framework, whereas `GraphGP` is a tool that can be used 

Numerous other packages implement the Vecchia approximation, though none are sufficiently performant for our intended use case in Milky Way tomography.

# Software design

`GraphGP` is a core component for building differentiable forward models. It is not an inference or modeling tool. It is designed to replace a dense Cholesky factorization with as little effort as possible, while still supporting low-level customization for advanced users. It is designed to be interoperable with any inference framework or other tools in the JAX ecosystem. We therefore adopt a minimal, functional API which allows users to apply GraphGP transformations to a user-provided graph. Most users will simply use GraphGP to generate the graph from an array of points, however it is just a tuple of arrays that advanced users can customize.

A key consideration is interoperability between the JAX implementation and custom CUDA extension. Users should be able to prototype in the JAX version and switch on the custom CUDA extension with a single keyword argument on machines that support it. Since the CUDA extension can only support discretized covariance functions, we require this format for the JAX version as well. There is essentially no performance impact and convenient helper functions are available for generating the required discretization.


# Research impact statement

`GraphGP` has been integrated into `NIFTy`.

JAX k-d tree has been used in halo-finding.

# AI usage disclosure
GitHub Copilot was used during development in auto-complete mode.

# References