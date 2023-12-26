

root_dir="$(git rev-parse --show-toplevel)"
cd "${root_dir}"

starcraft_path="${root_dir}/StarCraftII"
map_dir="${starcraft_path}/Maps"
printf "* Installing StarCraft 2 and SMACv2 maps into ${starcraft_path}\n"
cd "${root_dir}"
wget https://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
# The archive contains StarCraftII folder. Password comes from the documentation.
unzip -qo -P iagreetotheeula SC2.4.10.zip
mkdir -p "${map_dir}"
# Install Maps
wget https://github.com/oxwhirl/smacv2/releases/download/maps/SMAC_Maps.zip
unzip SMAC_Maps.zip
mkdir "${map_dir}/SMAC_Maps"
mv *.SC2Map "${map_dir}/SMAC_Maps"
printf "StarCraft II and SMAC are installed."

pip install git+https://github.com/oxwhirl/smacv2.git
