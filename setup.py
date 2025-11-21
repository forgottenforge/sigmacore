#!/usr/bin/env python3
"""
Sigma-C Setup Script
==========================================================
Copyright (c) 2025 ForgottenForge.xyz

Build and installation script for the Sigma-C Framework.

For commercial licensing without AGPL-3.0 obligations, contact:
[info@forgottenforge.xyz]

SPDX-License-Identifier: AGPL-3.0-or-later OR Commercial
"""
from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension, build_ext
import sys

# Define the C++ extension
ext_modules = [
    Pybind11Extension(
        "sigma_c_core",
        sources=[
            "sigma_c_core/src/susceptibility.cpp",
            "sigma_c_core/src/stats.cpp",
            "sigma_c_core/src/bindings.cpp",
        ],
        include_dirs=["sigma_c_core/include"],
        cxx_std=17,
        extra_compile_args=['/O2'] if sys.platform == 'win32' else ['-O3', '-march=native', '-fopenmp'],
    ),
]

setup(

    name="sigma_c_framework",
    version="1.2.3",
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    install_requires=[
        "numpy",
        "scipy",
        "pandas",
        "tqdm",
        "matplotlib",
        "seaborn",
        "requests",
        "yfinance",
        "pynvml",
        "scikit-learn",
        "pyyaml",
    ],
    entry_points={
        "console_scripts": [
            "sigma-c=sigma_c.cli:main",
        ],
    },
    zip_safe=False,
)
