---
title: Home
layout: default
use_math: false
---

This page collects the *Jupyter notebook* used for the graduate course on **Computational Methods for Imaging Science**, taught by Dr. Villa at Washington University in the Spring 2020 semester.

### hIPPYlib (Inverse Problems Python library)

The teaching material below uses [hIPPYlib](https://hippylib.github.io). hIPPYlib implements state-of-the-art scalable algorithms for PDE-based deterministic and Bayesian inverse problems.
It builds on [FEniCS](https://fenicsproject.org) (a parallel finite element element library) for the discretization of the partial differential equations and on [PETSc](https://www.mcs.anl.gov/petsc/)
for scalable and efficient linear algebra operations and solvers.


### A few important logistics:

- The teaching material consists of cloud-based interactive tutorials that mix instruction and theory with editable and runnable code. You can run the codes presented in class through your web browser. This will allow anyone to test our software and experiment with inverse problem algorithms quite easily, without running into installation issues or version discrepancies. Instructions to connect to the cloud resources, username and password will be posted on Canvas. Please do not exchange the user info.

- If you are curious to learn more about PDEs, finite element methods, and FEniCS, the fastest way to start learning this tool is to download and read the first chapter of the FEniCS book from here. Note the updated/new FEniCS tutorial version here. For more detailed instructions, please check out the ‘‘Getting started with FEniCS’’ document available [here](files/fenics_getting_started.pdf).

- For instructions on how to use Jupyter notebooks check out this [page](https://jupyter.readthedocs.io/en/latest/running.html#running).

### Teaching material

- Boundary Value Problems in FEniCS: [notebook](notebooks/Poisson.html), [recorded demonstration](https://wustl.box.com/s/5p8dvrxde5o6o6mvhak6t3odsoiz0yqq)

- Image denoising:
  - Tikhonov regularization: [notebook](notebooks/ImageDenoising_Tik.html), [recorded demonstration](https://wustl.box.com/s/w99ausfyconctd5a6mchx7rtm5y5j5dj)
  - Total Variation regularization: [notebook](notebooks/ImageDenoising_TV.html), [recorded demonstration](https://wustl.box.com/s/625hd84dpt493rcyap3y35wbakkdr329)
  - Primal-dual Total Variation regularization: [notebook](notebooks/ImageDenoising_PrimalDualTV.html), [recorded demonstration](https://wustl.box.com/s/zed8kxh34pcaq5ia4suqjjxom4moa6za)

- Constrained inverse problems:
  - Steepest descent: notebook, recorded demonstration
  - Inexact Newton Conjugate Gradient: notebook, recorded demonstration

- Analysis of the spectrum of the Hessian of inverse problem: notebook, recorded demonstration

- Bayesian solution of inverse problems:
  - Gaussian random fields: notebook, recorded demonstration
  - Sampling high-dimensional posterior distributions: notebook, recorded demonstration

### Acknowledgement

We would like to acknowledge the Extreme Science and Engineering Discovery Environment (XSEDE), which is supported by National Science Foundation grant ACI-1548562, for providing cloud computing resources (Jetstream) for this course through allocation TG-SEE190001.

hIPPYlib development is partially supported by National Science Foundation grants ACI-1550593 and ACI-1550547.
