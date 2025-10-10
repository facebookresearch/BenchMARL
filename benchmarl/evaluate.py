#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
import argparse
from pathlib import Path

from benchmarl.hydra_config import reload_experiment_from_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluates the experiment from a checkpoint file."
    )
    parser.add_argument(
        "checkpoint_file", type=str, help="The name of the checkpoint file"
    )
    args = parser.parse_args()
    checkpoint_file = str(Path(args.checkpoint_file).resolve())
    experiment = reload_experiment_from_file(checkpoint_file)

    experiment.config.evaluation_episodes = 1000 # OBS used to evaluate over more episodes. Does not overwrite the old config files! :)
    experiment.logger.calculate_extra = True
    experiment.evaluate()
    
    # if experiment.task.has_render(experiment.test_env) and experiment.config.render:
    #             video_frames = []

    #             def callback(env, td):
    #                 video_frames.append(
    #                     experiment.task.__class__.render_callback(experiment, env, td)
    #                 )

    # else:
    #     video_frames = None
    #     callback = None

    # if experiment.test_env.batch_size == ():
    #             rollouts = []
    #             for eval_episode in range(experiment.config.evaluation_episodes):
    #                 rollouts.append(
    #                     experiment.test_env.rollout(
    #                         max_steps=experiment.max_steps,
    #                         policy=experiment.policy,
    #                         callback=callback if eval_episode == 0 else None,
    #                         auto_cast_to_device=True,
    #                         break_when_any_done=True,
    #                     )
    #                 )
    # else:
    #     rollouts = experiment.test_env.rollout(
    #         max_steps=experiment.max_steps,
    #         policy=experiment.policy,
    #         callback=callback,
    #         auto_cast_to_device=True,
    #         break_when_any_done=False,
    #         # We are running vectorized evaluation we do not want it to stop when just one env is done
    #     )

    # print(rollouts)