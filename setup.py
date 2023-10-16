#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

from setuptools import find_packages, setup

setup(
    name="benchmarl",
    version="0.0.2",
    description="BenchMARL",
    url="https://github.com/facebookresearch/BenchMARL",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    packages=find_packages(),
    install_requires=["torchrl>=0.2.0", "tqdm", "hydra-core"],
    extras_require={
        "vmas": ["vmas>=1.2.10"],
        "pettingzoo": ["pettingzoo[all]>=1.24.1"],
    },
)
