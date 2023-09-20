
python -m pip install vmas
apt-get install -y python-opengl
apt-get install -y xvbf
apt-get install -y x11-utils
export DISPLAY=':99.0'
Xvfb :99 -screen 0 1400x900x24 > /dev/null 2>&1 &
