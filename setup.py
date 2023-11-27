#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import os
from pathlib import Path

from setuptools import find_packages, setup


def package_files(directory):
    paths = []
    for path, _, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths


extra_files = package_files(
    str(
        Path(os.path.dirname(os.path.realpath(__file__)))
        / Path("benchmarl")
        / Path("conf")
    )
)

setup(
    name="benchmarl",
    version="1.0.0",
    description="BenchMARL",
    url="https://github.com/facebookresearch/BenchMARL",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    install_requires=["torchrl>=0.2.0", "tqdm", "hydra-core"],
    extras_require={
        "vmas": ["vmas>=1.2.10"],
        "pettingzoo": ["pettingzoo[all]>=1.24.1"],
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={"": extra_files},
)
