# Feedback Arc Set: Research Workspace

> Raw research workspace and staging area, not a polished release. This repository holds exploratory code and generated experiment outputs from my feedback-arc-set (FAS) research: genetic and heuristic FAS algorithms, large-scale batch runs on NJIT's Wulver HPC cluster, and ranking outputs on connectome and other large graphs.

**Why it's public.** Later, cleaner packages pin specific commits of this workspace as provenance (for example, [mwfas-heuristic-metrics](https://github.com/SoroushVahidi/mwfas-heuristic-metrics) links its historical Wikimedia-derived artifacts here). For anything you want to use or cite, go to the documented repositories below.

## Polished repositories from this research line

| Repository | Paper |
|---|---|
| [ranking-by-feedback-arc-set](https://github.com/SoroushVahidi/ranking-by-feedback-arc-set) | *Training-free ranking from pairwise comparisons via acyclic graph construction*, The Journal of Supercomputing 2026, [doi:10.1007/s11227-026-08852-4](https://doi.org/10.1007/s11227-026-08852-4) |
| [minimum-weighted-fas-heuristics](https://github.com/SoroushVahidi/minimum-weighted-fas-heuristics) | *Incumbent-Protected SCC-Neighborhood Search for the Weighted FAS Problem*, SSRN preprint, [doi:10.2139/ssrn.6281222](https://doi.org/10.2139/ssrn.6281222) |
| [Hybrid-Interval-Based-Refinement-…](https://github.com/SoroushVahidi/Hybrid-Interval-Based-Refinement-for-Large-Scale-Weighted-Feedback-Arc-Set) | *Interval-Based Refinement Algorithms for Large-Scale Weighted FAS*, SSRN preprint, [doi:10.2139/ssrn.6221201](https://doi.org/10.2139/ssrn.6221201) |
| [minimum-feedback-challenge](https://github.com/SoroushVahidi/minimum-feedback-challenge) | *Feedforward Ordering in Neural Connectomes via Feedback Arc Minimization*, [arXiv:2506.13799](https://arxiv.org/abs/2506.13799) |
| [mwfas-heuristic-metrics](https://github.com/SoroushVahidi/mwfas-heuristic-metrics) | Audited MWFAS benchmark metrics (dataset package) |

## What's in this repo

- `codes/`: genetic and heuristic FAS implementations, HPC batch drivers (`feedback_two_heu_wulver-*.py`), notebooks.
- `datasets/`: generated ranking CSVs, log files and plots from those runs. These are large and not curated as a citable dataset release.

## Status

There is no license file in this repository, so all rights are reserved by default. No structured reproduction instructions are provided here. See the linked repositories above for documented, reproducible pipelines.
