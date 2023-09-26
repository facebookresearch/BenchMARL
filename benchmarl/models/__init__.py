from .gnn import GnnConfig
from .mlp import MlpConfig

model_config_registry = {"mlp": MlpConfig, "gnn": GnnConfig}
