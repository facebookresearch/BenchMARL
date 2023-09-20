
python -m pip install vmas
sudo apt-get update
sudo apt-get install python3-opengl xvfb
export DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
