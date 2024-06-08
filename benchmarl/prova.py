#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.
from benchmarl.environments import task_config_registry

if __name__ == "__main__":

    task_config_registry["meltingpot/chemistry__three_metabolic_cycles"].get_from_yaml()
