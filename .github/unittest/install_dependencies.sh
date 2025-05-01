

python -m pip install --upgrade pip
python -m pip install flake8 pytest pytest-cov hydra-core tqdm
python -m pip install torchrl
python -m pip install torch_geometric torchvision "av<14"

if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

cd ../BenchMARL
pip install -e .
