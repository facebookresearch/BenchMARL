# Different components for different groups

It is possible to use different algorithms and models for different agent groups.

In this folder, we provide examples on how to do this.

## Ensemble algorithm

Ensemble algorithms take as input a dictionary mapping group names to algorithm configs:

```pyhton
from benchmarl.algorithms import EnsembleAlgorithmConfig, IsacConfig, MaddpgConfig

algorithm_config = EnsembleAlgorithmConfig(
    {"agent": MaddpgConfig.get_from_yaml(), "adversary": IsacConfig.get_from_yaml()}
)
```

**Important: All algorithms need to be on-policy or off-policy, it is not possible to mix the two paradigms.**

## Ensemble model

Ensemble models take as input a dictionary mapping group names to model configs:

```pyhton
from benchmarl.models import EnsembleModelConfig, GnnConfig, MlpConfig

model_config = EnsembleModelConfig(
        {"agent": MlpConfig.get_from_yaml(), "adversary": GnnConfig.get_from_yaml()}
)
```

**Important: if you use ensemble models with sequence models, make sure the ensemble is the outer layer (you cannot make a
sequence of ensembles, but an ensemble of sequences yes).**
