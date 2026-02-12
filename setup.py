#!/usr/bin/env python
"""
3D Gaussian Splatting - Complete Implementation
A full-featured implementation of the 3D Gaussian Splatting paper
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gaussian-splatting",
    version="1.0.0",
    author="3DGS Implementation Team",
    description="Complete implementation of 3D Gaussian Splatting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/graphdeco-inria/gaussian-splatting",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch==2.0.0",
        "torchvision==0.15.0",
        "numpy==1.21.0",
        "pillow==9.0.0",
        "tqdm==4.65.0",
        "plyfile==0.8.0",
        "imageio==2.9.0",
        "imageio-ffmpeg==0.4.0",
        "lpips==0.1.4",
        "scikit-learn==1.0.0",
        "scipy==1.7.0",
        "tensorboard==2.10.0",
        "matplotlib==3.5.0",
        "trimesh==3.15.0",
        "open3d==0.16.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
        ],
    },
)
