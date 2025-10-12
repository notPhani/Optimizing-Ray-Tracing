# Optimizing-Ray-Tracing

A hybrid rendering pipeline that combines classical ray/path tracing with learned path generation and iterative refinement to improve sample efficiency and reduce variance on GPU-accelerated backends.

## Overview
- Core: a physically based renderer in PyTorch/CUDA for direct and multi-bounce lighting with modular samplers and materials.  
- ML: a conditional generator proposes candidate “path space” samples from compact scene descriptors, and a diffusion refiner iteratively steers paths toward higher-contribution regions under physics-aware constraints.  
- Goal: reduce variance and wall-clock per target error versus unguided sampling and classical path guiding baselines.

## ML architecture
- Conditional path prior (GAN): learns a distribution over candidate light paths conditioned on geometry/material/light features for importance sampling in high-payoff regions.  
- Diffusion refiner: starts from the generative prior and iteratively adjusts path samples toward optimal regions guided by rendering signals and feasibility constraints from the rendering equation.  
- Physics-informed objectives: radiance/image-domain losses, contribution-weighted rewards, and constraints that keep paths physically valid; optional RL-style reward shaping for iterative improvements.

## Training
- Data: curriculum from analytic scenes to complex scenes, with high-SPP references for supervision and ablation.  
- Losses: variance/MSE vs. references, contribution-weighted objectives, and regularizers enforcing physical plausibility of sampled paths.  
- Ablations: prior-only, diffusion-only, combined prior+diffusion; guided vs. unguided; MIS/path guiding baselines.

## Inference
- Encode scene → sample path prior → refine via diffusion steps → estimate radiance with classical estimator and MIS, targeting equal-quality images at fewer samples per pixel.

## Benchmarks
- Metrics: variance and MSE at equal SPP, PSNR/SSIM vs. high-SPP references, and wall-clock per attained error.  
- Reports: per-scene CSVs and image comparisons, with failures documented to guide model and sampler improvements.

## Roadmap
- Acceleration: BVH/quad-trees for intersection culling.  
- Effects: caustics, motion blur, extended BRDF/BSSRDF models.  
- Tooling: richer GPU CI, artifacted reports, model cards, and reproducible seeds.

<img src="./assets/snake.svg" alt="Contribution Snake" width="800">

