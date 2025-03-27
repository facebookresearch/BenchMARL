

python -m pip install --upgrade pip
python -m pip install flake8 pytest pytest-cov hydra-core tqdm torch torch_geometric torchvision "av<14"

if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

pip install torchrl

cd ../BenchMARL
pip install -e .
