from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch_geometric.nn import GATConv, global_mean_pool

from tsp_rl_kg.config import AgentModelConfig

torch_dtype = torch.float32


class VisionEncoder(nn.Module):
    """Plain torch vision encoder reusable outside any RL backend."""

    def __init__(self, observation_shape, vision_params, output_dim=96):
        super().__init__()

        self.num_conv_layers = vision_params.get("num_conv_layers", 4)
        self.conv_channels = vision_params.get("conv_channels", [64, 128, 256, 256])
        self.fc_dims = vision_params.get("fc_dims", [512])

        channels, _height, _width = observation_shape
        active_conv_channels = self.conv_channels[: self.num_conv_layers]
        if not active_conv_channels:
            raise ValueError("VisionEncoder requires at least one convolutional layer")

        conv_layers = []
        in_channels = channels
        for out_channels in active_conv_channels:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2, 2, ceil_mode=True))
            in_channels = out_channels

        conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        conv_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*conv_layers)

        last_channels = active_conv_channels[-1]
        total_conv_size = last_channels * 4 * 4

        fc_layers = []
        in_dim = total_conv_size
        for out_dim in self.fc_dims:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU())
            in_dim = out_dim

        fc_layers.append(nn.Linear(in_dim, output_dim))
        self.fc = nn.Sequential(*fc_layers)
        self.output_dim = output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.cnn(observations)
        return self.fc(x)


class GraphEncoder(nn.Module):
    """Plain torch geometric encoder reusable outside any RL backend."""

    def __init__(
        self,
        num_graph_node_features,
        graph_params,
        output_dim=96,
        gat_hidden_dim=48,
        num_edge_features=0,
    ):
        super().__init__()

        self.num_gat_layers = graph_params.get("num_gat_layers", 3)
        self.gat_heads = graph_params.get("gat_heads", [4, 2, 2])
        self.fc_dims = graph_params.get("fc_dims", [192])
        self.gat_hidden_dim = gat_hidden_dim

        active_heads = self.gat_heads[: self.num_gat_layers]
        if not active_heads:
            raise ValueError("GraphEncoder requires at least one GAT layer")

        gat_layers = []
        in_channels = num_graph_node_features
        edge_dim = num_edge_features if num_edge_features > 0 else None
        for heads in active_heads:
            out_channels = self.gat_hidden_dim * heads
            gat_layers.append(
                GATConv(
                    in_channels,
                    self.gat_hidden_dim,
                    heads=heads,
                    edge_dim=edge_dim,
                )
            )
            in_channels = out_channels

        self.gat = nn.ModuleList(gat_layers)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.fc_dims[0]),
            nn.ReLU(),
            nn.Linear(self.fc_dims[0], output_dim),
        )
        self.output_dim = output_dim

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for gat_layer in self.gat:
            x = F.relu(gat_layer(x, edge_index, edge_attr=edge_attr))

        x = global_mean_pool(x, batch)
        return self.fc(x)


class HybridEncoder(nn.Module):
    """Backend-neutral hybrid encoder for vision and graph observations."""

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 192,
        model_config: AgentModelConfig | None = None,
    ):
        super().__init__()

        if model_config is None:
            model_config = AgentModelConfig()

        if not {"num_nodes", "num_edges"}.issubset(observation_space.spaces):
            raise ValueError(
                "Observation schema v2 requires num_nodes/num_edges; fresh training is required"
            )

        self.disable_vision = model_config.disable_vision
        self.disable_graph = model_config.disable_graph
        self.vision_params = model_config.to_vision_params()
        self.graph_params = model_config.to_graph_params()

        first_fc_dim = self.vision_params["fc_dims"][-1] + self.graph_params["fc_dims"][-1]
        self.fc_dims = [first_fc_dim, first_fc_dim, first_fc_dim // 2, features_dim]
        self.dropout_p = model_config.dropout

        vision_shape = observation_space.spaces["vision"].shape
        num_node_features = observation_space.spaces["node_features"].shape[1]
        num_edge_features = observation_space.spaces["edge_attr"].shape[1]

        self.vision_processor = VisionEncoder(
            vision_shape,
            vision_params=self.vision_params,
            output_dim=features_dim,
        )
        self.graph_processor = GraphEncoder(
            num_node_features,
            graph_params=self.graph_params,
            output_dim=features_dim,
            gat_hidden_dim=model_config.gat_hidden_dim,
            num_edge_features=num_edge_features,
        )

        combined_input_size = self.vision_processor.output_dim + self.graph_processor.output_dim

        fc_layers = []
        in_dim = combined_input_size
        for out_dim in self.fc_dims:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU())
            in_dim = out_dim

        fc_layers.append(nn.Linear(in_dim, features_dim))
        self.fc = nn.Sequential(*fc_layers)
        self.dropout = nn.Dropout(p=self.dropout_p)

        self._initialize_weights()

    def _prepare_graph_batch(
        self,
        observations: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        nodes, edges, attrs, memberships = [], [], [], []
        offset = 0
        batch_size = observations["node_features"].shape[0]
        for i in range(batch_size):
            raw_n = observations["num_nodes"][i].item()
            raw_e = observations["num_edges"][i].item()
            if not (float(raw_n).is_integer() and float(raw_e).is_integer()):
                raise ValueError("Graph counts must be finite integers")
            n, e = int(raw_n), int(raw_e)
            if (
                not 1 <= n <= observations["node_features"].shape[1]
                or not 0 <= e <= observations["edge_attr"].shape[1]
            ):
                raise ValueError("Invalid graph counts for padded observation")
            x = observations["node_features"][i, :n].to(torch_dtype)
            raw_index = observations["edge_index"][i, :, :e]
            if not torch.isfinite(raw_index).all() or not torch.equal(
                raw_index, raw_index.long().to(raw_index.dtype)
            ):
                raise ValueError("Graph edge indices must be finite integers")
            index = raw_index.long()
            if e and (index.min() < 0 or index.max() >= n):
                raise ValueError("Graph edge references an invalid local node")
            nodes.append(x)
            edges.append(index + offset)
            attrs.append(observations["edge_attr"][i, :e].to(torch_dtype))
            memberships.append(torch.full((n,), i, device=x.device, dtype=torch.long))
            offset += n
        return torch.cat(nodes), torch.cat(edges, dim=1), torch.cat(memberships), torch.cat(attrs)

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        reference = observations["vision"]
        shape = (reference.shape[0], self.vision_processor.output_dim)
        vision_features = (
            reference.new_zeros(shape) if self.disable_vision else self.vision_processor(reference)
        )
        if self.disable_graph:
            graph_features = reference.new_zeros(
                (reference.shape[0], self.graph_processor.output_dim)
            )
        else:
            x, edge_index, batch, edge_attr = self._prepare_graph_batch(observations)
            graph_features = self.graph_processor(x, edge_index, batch, edge_attr=edge_attr)

        combined = torch.cat((vision_features, graph_features), dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def sanity_check(self, observations: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            output = self.forward(observations)
            logger.info(f"Output shape: {output.shape}")
            logger.info(f"Output mean: {output.mean().item():.4f}")
            logger.info(f"Output std: {output.std().item():.4f}")
            vision_mean = self.vision_processor(observations["vision"]).mean().item()
            logger.info(f"Vision features mean: {vision_mean:.4f}")
            x, edge_index, batch, edge_attr = self._prepare_graph_batch(observations)
            graph_mean = (
                self.graph_processor(
                    x,
                    edge_index,
                    batch,
                    edge_attr=edge_attr,
                )
                .mean()
                .item()
            )
            logger.info(f"Graph features mean: {graph_mean:.4f}")
