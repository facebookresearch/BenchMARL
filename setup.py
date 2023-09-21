from setuptools import find_packages, setup

setup(
    name="benchmarl",
    version="0.0.1",
    description="BenchMARL",
    url="https://github.com/facebookresearch/BenchMARL",
    author="Matteo Bettini",
    author_email="mb2389@cl.cam.ac.uk",
    packages=find_packages(),
    install_requires=["torchrl", "tqdm"],
    include_package_data=True,
)
