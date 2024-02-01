

python -m pip install --upgrade pip
python -m pip install flake8 pytest pytest-cov hydra-core tqdm torch_geometric

if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

python -m pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall

pip install torchrl

cd ../BenchMARL
pip install -e .
