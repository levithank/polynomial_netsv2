# Polynomial Networks on Synthetic Manifolds
### Probing product-polynomial (Π-net) models with low-dimensional toy data

> Forked from [grigorisg9gr/polynomial_nets](https://github.com/grigorisg9gr/polynomial_nets) — the official implementation of ["Π-nets: Deep Polynomial Neural Networks"](https://openaccess.thecvf.com/content_CVPR_2020/papers/Chrysos_P-nets_Deep_Polynomial_Neural_Networks_CVPR_2020_paper.pdf) (CVPR'20) and its [T-PAMI'21 extension](https://arxiv.org/abs/2006.13026). This fork adapts that architecture to small, controllable synthetic datasets to study **why** and **when** polynomial networks capture the geometry of the data.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![ArXiv](https://img.shields.io/badge/Preprint-ArXiv-blue.svg)](https://arxiv.org/abs/2006.13026)
[![BlogPost](https://img.shields.io/badge/BlogPost-site-red.svg)](https://grigorisg9gr.github.io/polynomial-nets/)

## Overview

Π-nets model their output as a high-degree **polynomial** of the input. Because a dense polynomial expansion is intractable, the coefficients are captured with a tensor decomposition — here, **CP (CANDECOMP/PARAFAC) decomposition** — which factors the polynomial into a product of simpler blocks (a *product of polynomials*). See the upstream repo and paper for the full formulation.

This fork keeps that architecture but changes the data. Instead of images, it fits product-polynomial models to **low-dimensional synthetic manifolds** whose structure is known in advance. The motivation is the **manifold hypothesis**: real data is assumed to lie on or near a low-dimensional manifold embedded in a higher-dimensional space. Toy datasets make that structure explicit and controllable — you set the intrinsic dimension and geometry yourself — which turns them into a clean testbed for questions that are hard to isolate on real data:

- **What does the model actually learn** about the underlying manifold?
- **How many latent dimensions** does the product-polynomial model use to represent it, and does that match the true intrinsic dimension?
- **How does behavior change with polynomial degree and CP-decomposition structure** — where does more expressivity help, and where does the model become unstable or fail?

## Experiments

Each folder is a self-contained experiment; follow the instructions inside it to run and reproduce.

- **`toy_datasets/`** — the core of this fork. Product-polynomial models fit to low-dimensional synthetic manifolds:
  - **`circle`** — points on a 1-D circle in 2-D; the simplest closed manifold.
  - **`spiral`** — a curved 1-D manifold (and the classic two-spiral separation) in 2-D; tests nonlinearity and curvature.
  - **`swiss_roll`** — a 2-D sheet rolled into 3-D; the canonical manifold-hypothesis test, intrinsically 2-D but embedded in 3-D.
- **`image_generation_chainer/`** — inherited from upstream. Reproduces the unconditional **Fashion-MNIST** generator — a product-of-polynomials GAN *without* activation functions between layers — in the Chainer framework.

<!-- TODO: rename the folders above to match your actual directory layout. -->

## Results

**Work in progress.** Figures and findings will land here as the experiments finalize. Planned outputs:

- Learned vs. ground-truth manifold for each toy dataset.
- How reconstruction quality and recovered latent dimensionality shift with **polynomial degree** and **CP-decomposition rank**.
- The regime where increasing degree stops helping and the model destabilizes.

<!-- TODO: drop in your plots (e.g. figures/...) and a one-line takeaway under each. -->

## About Π-nets (upstream)

Π-nets don't rely on a single architecture — the network is defined by the recursive formula that constructs it, so many architectures can be built from the same idea. The evaluation in the original paper shows product-polynomial models can match or improve strong baselines on tasks including image generation, face recognition, and mesh representation learning. Full details, pretrained experiments (MXNet / PyTorch / Chainer), and a one-minute video pitch are in the [upstream repository](https://github.com/grigorisg9gr/polynomial_nets).

## Citing

This fork builds directly on the Π-nets work. If you use this code, please cite the original papers:

```bibtex
@inproceedings{poly2020,
  title={$\Pi-$nets: Deep Polynomial Neural Networks},
  author={Chrysos, Grigorios and Moschoglou, Stylianos and Bouritsas, Giorgos and Panagakis, Yannis and Deng, Jiankang and Zafeiriou, Stefanos},
  booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={7325--7335},
  year={2020}
}
```

```bibtex
@article{poly2021,
  author={Chrysos, Grigorios and Moschoglou, Stylianos and Bouritsas, Giorgos and Deng, Jiankang and Panagakis, Yannis and Zafeiriou, Stefanos},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={Deep Polynomial Neural Networks},
  volume={44},
  number={8},
  pages={4021--4034},
  year={2021},
  doi={10.1109/TPAMI.2021.3058891}
}
```

## References

[1] Grigorios G. Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Yannis Panagakis, Jiankang Deng and Stefanos Zafeiriou, **Π-nets: Deep Polynomial Neural Networks**, *Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020.

[2] Grigorios G. Chrysos, Stylianos Moschoglou, Giorgos Bouritsas, Jiankang Deng, Yannis Panagakis and Stefanos Zafeiriou, **Deep Polynomial Neural Networks**, *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 2021.

## License & attribution

This is a research fork. All credit for the original architecture and implementation goes to the Π-nets authors. The upstream code is released under **CC BY-NC 4.0** (attribution, non-commercial), and this fork inherits the same terms — see the [license](https://creativecommons.org/licenses/by-nc/4.0/).