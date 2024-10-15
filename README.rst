
Benchmark for MRI reconstruction algorithms
===========================================
|Build Status| |Python 3.6+|

Benchopt is a package to simplify and make more transparent and
reproducible the comparisons of optimization algorithms.
This benchmark is dedicated to **MRI PnP reconstruction**.

Details explanation are available in the following paper: XXX

Install
--------

This benchmark can be run using the following commands:

.. code-block::

   $ pip install -U benchopt
   $ git clone https://github.com/paquiteau/benchmark_mri_pnp
   $ benchopt run benchmark_mri_reconstruction

Apart from the problem, options can be passed to ``benchopt run``, to restrict the benchmarks to some solvers or datasets, e.g.:

.. code-block::

	$ benchopt run benchmark_mri_reconstruction -s solver1 -d dataset2 --max-runs 10 --n-repetitions 10


Use ``benchopt run -h`` for more details about these options, or visit https://benchopt.github.io/api.html.

.. |Build Status| image:: https://github.com/paquiteau/benchmark_mri_pnp/workflows/Tests/badge.svg
   :target: https://github.com/bmalezieux/benchmark_mri_pnp/actions
.. |Python 3.9+| image:: https://img.shields.io/badge/python-3.9%2B-blue
   :target: https://www.python.org/downloads/release/python-390/


Description of the benchmark
----------------------------
This benchmark focuses on iterative reconstruction methods for 2D - Multicoil Non Cartesian MRI using Plug and Play Methods.


Dataset
~~~~~~~
FastMRI MultiCoil Test set *needs to be dowloaded separately.*

Objective
~~~~~~~~~
We compute the PSNR and SSIM

Solvers
~~~~~~~
- Compressed Sensing (Fista with Wavelet)
- Unrolled Network (NCPDNET)
- HQS Preconditioned (ours)
- PnP Preconditioned ?
-
