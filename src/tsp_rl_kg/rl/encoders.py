"""Compact, deterministic policy encoders with isolated graph batching."""

import math

import torch
from torch import nn
from torch_geometric.nn import GATConv
from torch_geometric.utils import softmax

from tsp_rl_kg.config import AgentModelConfig


class VisionEncoder(nn.Module):
    def __init__(self, observation_shape, vision_params, output_dim=64):
        super().__init__()
        channels, height, width = observation_shape
        layers = []
        for output in vision_params["conv_channels"][: vision_params["num_conv_layers"]]:
            layers.extend([nn.Conv2d(channels, output, 3, padding=1), nn.ReLU()])
            channels = output
        self.cnn = nn.Sequential(*layers, nn.Flatten())
        self.fc = nn.Linear(channels * height * width, output_dim)
        self.output_dim = output_dim

    def forward(self, observations):
        return self.fc(self.cnn(observations.float()))


class GraphEncoder(nn.Module):
    def __init__(
        self,
        num_graph_node_features,
        graph_params,
        output_dim=64,
        gat_hidden_dim=32,
        num_edge_features=0,
    ):
        super().__init__()
        self.gat = nn.ModuleList()
        width = num_graph_node_features
        for heads in graph_params["gat_heads"][: graph_params["num_gat_layers"]]:
            self.gat.append(
                GATConv(
                    width,
                    gat_hidden_dim,
                    heads=heads,
                    edge_dim=num_edge_features or None,
                    dropout=0,
                )
            )
            width = gat_hidden_dim * heads
        # Include raw relative geometry so it cannot be washed out by local aggregation.
        width += 5
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.fc = nn.Sequential(nn.Linear(width * 2, output_dim), nn.ReLU())
        self.output_dim = output_dim

    def forward(self, x, edge_index, batch, edge_attr=None):
        geometry = x[:, -5:]
        players = geometry[:, 2] > 0.5
        for layer in self.gat:
            x = torch.relu(layer(x, edge_index, edge_attr=edge_attr))
        x = torch.cat([x, geometry], dim=1)
        # Exactly one explicitly marked player in each sample.
        anchors = x[players]
        if not torch.equal(
            torch.bincount(batch[players], minlength=int(batch.max()) + 1),
            torch.ones(int(batch.max()) + 1, device=batch.device, dtype=torch.long),
        ):
            raise ValueError("Graph must contain one player per sample")
        scores = (self.key(x) * self.query(anchors)[batch]).sum(-1) / math.sqrt(x.shape[1])
        weights = softmax(scores, batch)
        pooled = torch.zeros_like(anchors).index_add(0, batch, weights[:, None] * x)
        return self.fc(torch.cat([anchors, pooled], dim=1))


class HybridEncoder(nn.Module):
    def __init__(self, observation_space, features_dim=128, model_config=None):
        super().__init__()
        config = model_config or AgentModelConfig()
        required = {"num_nodes", "num_edges", "status", "path_queries"}
        if not required.issubset(observation_space.spaces):
            raise ValueError("Observation schema v4 required; fresh training is required")
        self.disable_vision, self.disable_graph = config.disable_vision, config.disable_graph
        self.use_path_queries = config.use_path_queries and not self.disable_graph
        self.vision_params, self.graph_params = config.to_vision_params(), config.to_graph_params()
        self.fc_dims = config.fusion_dims
        self.dropout_p = 0
        self.dropout = nn.Identity()
        self.vision_processor = (
            None
            if self.disable_vision
            else VisionEncoder(observation_space["vision"].shape, self.vision_params, 64)
        )
        self.graph_processor = (
            None
            if self.disable_graph
            else GraphEncoder(
                observation_space["node_features"].shape[1],
                self.graph_params,
                64,
                config.gat_hidden_dim,
                observation_space["edge_attr"].shape[1],
            )
        )
        self.status_processor = nn.Sequential(
            nn.Linear(observation_space["status"].shape[0], 32), nn.ReLU()
        )
        self.query_processor = (
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(math.prod(observation_space["path_queries"].shape), 32),
                nn.ReLU(),
            )
            if self.use_path_queries
            else None
        )
        width = (
            32
            + 64 * (not self.disable_vision)
            + 64 * (not self.disable_graph)
            + 32 * self.use_path_queries
        )
        layers = []
        for output in [*config.fusion_dims, features_dim]:
            layers.extend([nn.Linear(width, output), nn.ReLU()])
            width = output
        self.fc = nn.Sequential(*layers)
        self._initialize_weights()

    def _prepare_graph_batch(self, observations):
        nodes = observations["node_features"]
        counts = observations["num_nodes"].reshape(-1)
        edge_counts = observations["num_edges"].reshape(-1)
        raw_edges = observations["edge_index"]
        if not (
            torch.isfinite(counts).all()
            and torch.isfinite(edge_counts).all()
            and (counts == counts.long()).all()
            and (edge_counts == edge_counts.long()).all()
            and (counts >= 1).all()
            and (counts <= nodes.shape[1]).all()
            and (edge_counts >= 0).all()
            and (edge_counts <= raw_edges.shape[2]).all()
        ):
            raise ValueError("Invalid graph counts for padded observation")
        counts, edge_counts = counts.long(), edge_counts.long()
        node_mask = torch.arange(nodes.shape[1], device=nodes.device)[None, :] < counts[:, None]
        edge_mask = (
            torch.arange(raw_edges.shape[2], device=nodes.device)[None, :] < edge_counts[:, None]
        )
        real_edges = raw_edges.permute(0, 2, 1)[edge_mask]
        edge_members = torch.arange(len(counts), device=nodes.device)[:, None].expand_as(edge_mask)[
            edge_mask
        ]
        if not (
            torch.isfinite(real_edges).all()
            and (real_edges == real_edges.long()).all()
            and (real_edges >= 0).all()
            and (real_edges < counts[edge_members, None]).all()
        ):
            raise ValueError("Graph edge references an invalid local node")
        offsets = counts.cumsum(0) - counts
        edges = (real_edges.long() + offsets[edge_members, None]).T
        memberships = torch.arange(len(counts), device=nodes.device)[:, None].expand_as(node_mask)[
            node_mask
        ]
        return (
            nodes[node_mask].float(),
            edges,
            memberships,
            observations["edge_attr"][edge_mask].float(),
        )

    def forward(self, observations):
        features = [self.status_processor(observations["status"].float())]
        if not self.disable_vision:
            features.append(self.vision_processor(observations["vision"]))
        if not self.disable_graph:
            x, edges, batch, attrs = self._prepare_graph_batch(observations)
            features.append(self.graph_processor(x, edges, batch, attrs))
        if self.use_path_queries:
            features.append(self.query_processor(observations["path_queries"].float()))
        return self.fc(torch.cat(features, dim=1))

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def sanity_check(self, observations):
        if not torch.isfinite(self(observations)).all():
            raise ValueError("Nonfinite policy features")
