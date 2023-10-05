python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.n_iters=3 experiment.checkpoint_interval=1
python benchmarl/run.py task=vmas/balance algorithm=mappo experiment.n_iters=6  experiment.restore_file="/hydra/experiment/folder/checkpoint/checkpoint_03.pt"
