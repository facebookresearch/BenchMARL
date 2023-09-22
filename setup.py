from setuptools import find_packages, setup

setup(
    name="benchmarl",
    version="0.0.1",
    description="BenchMARL",
    url="https://github.com/facebookresearch/BenchMARL",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    packages=find_packages(),
    install_requires=["torchrl", "tqdm", "hydra-core"],
    extras_require={
        "tasks": ["vmas>=1.2.10", "pettingzoo[all]>=1.24.1"],
    },
    include_package_data=True,
)
