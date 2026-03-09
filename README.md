# Differential Gaussian Rasterization with Importance-based Pruning

Modified CUDA rasterizer for 3D Gaussian Splatting with weighted importance scoring for pruning.

## Features

- **Gaussian Counting**: Track how many times each Gaussian contributes to rendering
- **Importance Scoring**: Compute opacity-based importance scores
- **Weighted Importance**: Compute alpha × T weighted scores for more accurate pruning decisions

## Based On

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original implementation
- [LightGaussian](https://github.com/VITA-Group/LightGaussian) - Importance-based pruning method
