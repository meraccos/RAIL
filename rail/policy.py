import torch
from stable_baselines3.common.policies import ActorCriticPolicy


class CustomPPOPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_actions_from_features(
        self, deterministic: bool = False, features=None
    ) -> torch.Tensor:
        (latent_pi, _) = self.mlp_extractor(features)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions.reshape((-1, *self.action_space.shape))
