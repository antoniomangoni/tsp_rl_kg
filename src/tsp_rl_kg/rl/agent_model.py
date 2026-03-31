import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from tsp_rl_kg.config import AgentModelConfig
from tsp_rl_kg.rl.encoders import GraphEncoder, HybridEncoder, VisionEncoder


class AgentModel(BaseFeaturesExtractor):
    """Stable-Baselines3 adapter over the backend-neutral hybrid encoder core."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 192,
        model_config: AgentModelConfig | None = None,
    ):
        super().__init__(observation_space, features_dim=features_dim)
        self.encoder = HybridEncoder(
            observation_space=observation_space,
            features_dim=features_dim,
            model_config=model_config,
        )
        self.disable_vision = self.encoder.disable_vision
        self.disable_graph = self.encoder.disable_graph
        self.vision_params = self.encoder.vision_params
        self.graph_params = self.encoder.graph_params
        self.fc_dims = self.encoder.fc_dims
        self.dropout_p = self.encoder.dropout_p

    @property
    def vision_processor(self) -> VisionEncoder:
        return self.encoder.vision_processor

    @property
    def graph_processor(self) -> GraphEncoder:
        return self.encoder.graph_processor

    @property
    def fc(self):
        return self.encoder.fc

    @property
    def dropout(self):
        return self.encoder.dropout

    def forward(self, observations):
        return self.encoder(observations)

    def _initialize_weights(self) -> None:
        self.encoder._initialize_weights()

    def sanity_check(self, observations) -> None:
        self.encoder.sanity_check(observations)


VisionProcessor = VisionEncoder
GraphProcessor = GraphEncoder
HybridEncoderCore = HybridEncoder
