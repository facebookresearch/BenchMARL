
python -m pip install vmas
sudo apt-get update
sudo apt-get install --fix-missing libgl1-mesa-dev
sudo apt-get install -y python-opengl
sudo apt-get install -y xvbf
sudo apt-get install -y x11-utils
export DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
