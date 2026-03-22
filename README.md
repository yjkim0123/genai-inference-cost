# Predictive Modeling of Generative AI Inference Cost

This repository contains the data and code for the paper:

**"Predictive Modeling of Generative AI Inference Cost Across Hardware Architectures"**

## Dataset
- 4,133 energy measurements across 7 generative models and 3 hardware platforms
- Models: Phi-3, Mistral-7B, MusicGen (small/medium), SD v1.5, SDXL, AnimateDiff
- Hardware: Apple M4 Pro, NVIDIA A100, NVIDIA H100

## Key Findings
- Within-platform prediction: R² = 0.923 (SVR with log features)
- Cross-platform prediction: R² < 0.1 (LOHO-CV)
- AR models are hardware-bound (TDP dominates)
- Diffusion models are workload-bound (output complexity dominates)

## Citation
If you use this data or code, please cite our paper.
