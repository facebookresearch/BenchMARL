

python -m pip install --upgrade pip
python -m pip install flake8 pytest pytest-cov hydra-core tqdm torch_geometric

if [ -f requirements.txt ]; then pip install -r requirements.txt; fi

python -m pip install torch torchvision "av<14"
# Not using nightly torch
# python -m pip install --pre torch --extra-index-url https://download.pytorch.org/whl/nightly/cpu --force-reinstall

cd ../BenchMARL
pip install -e .
pip uninstall --yes torchrl
pip uninstall --yes tensordict

cd ..

git clone https://github.com/pytorch/tensordict.git
cd tensordict
python setup.py develop
cd ..
git clone https://github.com/pytorch/rl.git
cd rl
python setup.py develop
