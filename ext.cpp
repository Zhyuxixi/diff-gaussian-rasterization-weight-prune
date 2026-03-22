/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <torch/extension.h>
#include "rasterize_points.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // Keep pybind registrations aligned with the current rasterizer signatures,
  // including contrib-threshold and contrib-max forwarding parameters.
  m.def("rasterize_gaussians", &RasterizeGaussiansCUDA);
  m.def("rasterize_gaussians_backward", &RasterizeGaussiansBackwardCUDA);
  m.def("mark_visible", &markVisible);
  m.def("count_gaussians", &CountGaussiansCUDA);
  m.def("count_gaussians_weighted", &CountGaussiansWeightedCUDA);
  m.def("count_gaussians_weighted_residual", &CountGaussiansWeightedResidualCUDA);
}
