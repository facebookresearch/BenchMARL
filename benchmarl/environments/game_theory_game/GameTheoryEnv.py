import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.envs import EnvBase
from torchrl.data import Bounded, Composite, UnboundedContinuous, Categorical
from torchrl.envs.utils import check_env_specs
from typing import Optional, Tuple, Dict, Any
from benchmarl.utils import DEVICE_TYPING

class TwoPlayerGameTheoryEnv(EnvBase):
    """
    A simple two-player game theory environment.
    Supports 4 different games: Prisoner's Dilemma, Stag Hunt, Chicken, and Battle of Sexes
    """
    
    # Game payoff matrices (row player = agent 0, column player = agent 1)
    # Each payoff is (reward_agent0, reward_agent1)
    GAMES = {
        "prisoners_dilemma": {
            "payoffs": {
                (0, 0): (-1, -1),  # Both cooperate
                (0, 1): (-3, 0),   # Cooperate vs Defect
                (1, 0): (0, -3),   # Defect vs Cooperate
                (1, 1): (-2, -2),  # Both defect
            },
            "action_names": {0: "Cooperate", 1: "Defect"}
        },
        "stag_hunt": {
            "payoffs": {
                (0, 0): (4, 4),    # Both hunt stag
                (0, 1): (0, 2),    # Stag vs Hare
                (1, 0): (2, 0),    # Hare vs Stag
                (1, 1): (1, 1),    # Both hunt hare
            },
            "action_names": {0: "Hunt Stag", 1: "Hunt Hare"}
        },
        "chicken": {
            "payoffs": {
                (0, 0): (0, 0),    # Both Swerve
                (0, 1): (-1, 1),   # Swerve vs Straight
                (1, 0): (1, -1),   # Straight vs Swerve
                (1, 1): (-10, -10),# Both Straight (crash)
            },
            "action_names": {0: "Swerve", 1: "Straight"}
        },
        "battle_of_sexes": {
            "payoffs": {
                (0, 0): (3, 2),    # Both choose Opera
                (0, 1): (0, 0),    # Opera vs Football
                (1, 0): (0, 0),    # Football vs Opera
                (1, 1): (2, 3),    # Both choose Football
            },
            "action_names": {0: "Opera", 1: "Football"}
        }
    }
    
    n_agents = 2
    
    def __init__(
        self, 
        game_name: str = "prisoners_dilemma",
        seed=None,
        device: DEVICE_TYPING = "cpu"
    ):
        super().__init__(device=device)
        
        if game_name not in self.GAMES:
            raise ValueError(f"Game must be one of {list(self.GAMES.keys())}")
        
        self.game_name  = game_name
        self.game_data  = self.GAMES[game_name]
        self.env_device = device
        self._make_spec()
        self._set_seed(seed)
        
        self.group_map = {'agents': ['agent 0', 'agent 1']}
    
    def _make_spec(self):
        # Action spec: 2 agents, each with 2 discrete actions
        self.action_spec = Composite(
            agents=Composite(
                action=Categorical(
                    n=2,
                    shape=torch.Size((self.n_agents,)),
                    device=self.env_device,
                    dtype=torch.int8
                ), 
            shape=torch.Size((self.n_agents,))
            )
        )
        
        # Observation spec: simple observation (can be expanded)
        self.observation_spec = Composite(
            agents=Composite(
                state=Bounded(
                    low=0, high=0, 
                    shape=torch.Size((self.n_agents, 1)),
                    device=self.device,
                    dtype=torch.float,
                ),
                shape=torch.Size((self.n_agents,))
            )
        )
        # since the environment is stateless, we expect the previous output as input.
        # For this, ``EnvBase`` expects some state_spec to be available
        self.state_spec = self.observation_spec.clone()
        
        # Reward spec for two agents
        self.reward_spec = Composite(
            agents=Composite(
                reward=UnboundedContinuous(
                    shape=torch.Size((self.n_agents, )),
                    device=self.env_device,
                    dtype=torch.float
                ),
                shape=torch.Size((self.n_agents,))
            )
        )
        
        # Done spec
        self.done_spec = Composite(
            done=Bounded(
                low=0, high=1,
                shape=torch.Size((1,)),
                device=self.env_device,
                dtype=torch.bool
            ),
            terminated=Bounded(
                low=0, high=1,
                shape=torch.Size((1,)),
                device=self.env_device,
                dtype=torch.bool
            ),
            truncated=Bounded(
                low=0, high=1,
                shape=torch.Size((1,)),
                device=self.env_device,
                dtype=torch.bool
            ),
        )
        
    def _step(self, tensordict: TensorDict) -> TensorDict:
        """Execute one step. Input must contain actions for both agents."""
        actions = tensordict['agents', "action"]  # Shape: (batch, 2, 1)

        # Compute payoffs for each batch element
        a0 = int(actions[..., 0].item())
        a1 = int(actions[..., 1].item())
        payoffs = self.game_data["payoffs"][(a0, a1)]

        # rewards shape must be (batch, n_agents, 1) to match reward_spec
        rewards = torch.tensor(payoffs, device=self.device, dtype=torch.float).unsqueeze(-1)

        # Create next tensordict with the same batch size as the input
        done = torch.ones(1, dtype=torch.bool, device=self.device)
        terminated = torch.ones(1, dtype=torch.bool, device=self.device)
        truncated = torch.zeros(1, dtype=torch.bool, device=self.device)

        next_td = TensorDict(
            {
                "agents": TensorDict(
                    {"reward": rewards,
                     'state': tensordict['agents', 'state']},
                    batch_size=torch.Size([self.n_agents]),
                    device=self.device,
                ),
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
            },
            device=self.device,
        )

        return next_td
    
    def _reset(self, tensordict: TensorDict, **kwargs) -> TensorDict:
        out = TensorDict(
            {
                "done": torch.zeros(1, dtype=torch.bool, device=self.device),
                "terminated": torch.zeros(1, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(1, dtype=torch.bool, device=self.device),
            },
            device=self.device,
        )
        # Add dummy observations
        out.update(self.observation_spec.rand())
        return out
    
    def _set_seed(self, seed: Optional[int]):
        """Set seed for reproducibility"""
        if seed is not None:
            torch.manual_seed(seed)
    
    def get_payoff(self, action0: int, action1: int) -> Tuple[int, int]:
        """Get payoff for given action pair"""
        return self.game_data["payoffs"][(action0, action1)]
    
    def get_action_names(self) -> Dict[int, str]:
        """Get action names for current game"""
        return self.game_data["action_names"]
    
    def print_the_game(self) -> None:
        """Simple text rendering"""
        print(f"\n=== {self.game_name.upper().replace('_', ' ')} ===")
        print(f"Action 0: {self.game_data['action_names'][0]}")
        print(f"Action 1: {self.game_data['action_names'][1]}")
        print("\nPayoff Matrix (Agent 0, Agent 1):")
        print(f"  {self.game_data['action_names'][0]:12} {self.game_data['action_names'][1]:12}")
        for a0 in [0, 1]:
            row = f"{self.game_data['action_names'][a0]:12}"
            for a1 in [0, 1]:
                payoff = self.game_data["payoffs"][(a0, a1)]
                row += f" ({payoff[0]:2d}, {payoff[1]:2d})  "
            print(row)
        print()
        
    def get_rand_action(self, tensordict: TensorDictBase | None = None) -> TensorDict:
        if tensordict is None:
            tensordict = self.reset()
        tensordict = tensordict.update(self.action_spec.rand())
        return tensordict


# Example usage and testing
def test_environment():
    """Test all four game configurations"""
    
    games = list(TwoPlayerGameTheoryEnv.GAMES.keys())
    
    for game_name in games:
        print(f"\n{'='*50}")
        print(f"Testing {game_name}")
        print('='*50)
        
        # Create environment
        env = TwoPlayerGameTheoryEnv(game_name=game_name)
        env.print_the_game()
        
        # Reset environment
        td = env.reset()
        print(f"Reset state shape: {td.shape}")
        
        # Test different action combinations
        test_actions = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.long)
        
        for actions in test_actions:
            td = None
            td = env.get_rand_action(td)
            td['agents', "action"][0] = actions[0] # Agent 0
            td['agents', "action"][1] = actions[1] # Agent 1
            td = env.step(td)

            print(f"\nActions: {actions.tolist()}")
            print(f"Rewards: {td['next', 'agents', 'reward'].tolist()}")
            print(f"Done: {td['next', 'done'].tolist()}")

if __name__ == "__main__":
    check_env_specs(TwoPlayerGameTheoryEnv())
    
    # Test basic functionality
    # test_environment()