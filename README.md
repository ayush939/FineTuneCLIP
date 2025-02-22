# SAFT (unofficial implementation)

This repository is an unofficial PyTorch implementaion of [SAFT](https://arxiv.org/abs/2407.03036) Sparse Adaptation for Fine Tuning. This is a type of a PEFT method that fine tunes large models on the in-distribution dataset in such a way that the model does not lose its generalization on related datasets. This approach is architecture independent.

The repository makes an effort to reproduce results for fine-tuning CLIP (with transformer backbone) using SAFT on ImageNet and compares the generalization on the dataset ImageNet-V2, ImageNet-S, ImageNet-A and ImageNet-R.