#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from pathlib import Path

import torch
from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
os.path.dirname(os.path.abspath(__file__))


def _matching_cuda_package_dirs(package_prefix, child_parts, sentinel):
    cuda_version = (torch.version.cuda or "").strip()
    candidates = []
    conda_prefix = os.environ.get("CONDA_PREFIX")
    roots = []
    if conda_prefix:
        prefix_path = Path(conda_prefix)
        if len(prefix_path.parents) >= 2:
            roots.append(prefix_path.parents[1] / "pkgs")
    roots.append(Path.home() / ".conda" / "pkgs")

    for root in roots:
        if not root.exists():
            continue
        for package_dir in sorted(root.glob(f"{package_prefix}{cuda_version}*")):
            child_dir = package_dir.joinpath(*child_parts)
            if (child_dir / sentinel).exists():
                candidates.append(str(child_dir))
    return candidates


def _ensure_cuda_home_from_conda_cache():
    if cpp_extension.CUDA_HOME:
        return
    cuda_version = (torch.version.cuda or "").strip()
    if not cuda_version:
        return

    conda_prefix = os.environ.get("CONDA_PREFIX")
    roots = []
    if conda_prefix:
        prefix_path = Path(conda_prefix)
        if len(prefix_path.parents) >= 2:
            roots.append(prefix_path.parents[1] / "pkgs")
    roots.append(Path.home() / ".conda" / "pkgs")

    for root in roots:
        if not root.exists():
            continue
        for package_dir in sorted(root.glob(f"cuda-nvcc-{cuda_version}*")):
            if (package_dir / "bin" / "nvcc.exe").exists():
                os.environ["CUDA_HOME"] = str(package_dir)
                cpp_extension.CUDA_HOME = str(package_dir)
                return


_ensure_cuda_home_from_conda_cache()

setup(
    name="diff_gaussian_rasterization_w_prune",
    packages=['diff_gaussian_rasterization_w_prune'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization_w_prune._C",
            include_dirs=_matching_cuda_package_dirs(
                "cuda-cudart-dev-", ["include"], "cuda_runtime.h"
            ) + _matching_cuda_package_dirs(
                "cuda-cccl-", ["include"], "cub/cub.cuh"
            ),
            library_dirs=_matching_cuda_package_dirs(
                "cuda-cudart-", ["lib", "x64"], "cudart.lib"
            ),
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cpp",
            "ext.cpp"],
            extra_compile_args={"nvcc": [
                "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/"),
                "-allow-unsupported-compiler",
                "-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH",
            ]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
