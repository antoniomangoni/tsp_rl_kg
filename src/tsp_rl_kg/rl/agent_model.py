import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

from tsp_rl_kg.config import AgentModelConfig

torch_dtype = torch.float32


class VisionProcessor(BaseFeaturesExtractor):
    """
    Neural network module for processing visual input (image data).

    Consists of configurable convolutional layers followed by fully
    connected layers, extracting hierarchical features from image
    observations and outputting a fixed-size feature vector.

    Attributes
    ----------
    num_conv_layers : int
        Number of convolutional layers (default: 4).
    conv_channels : list of int
        Output channels for each convolutional layer.
        Default: [32, 64, 128, 256].
    fc_dims : list of int
        Sizes of fully connected layers after convolutions.
        Default: [512].
    cnn : nn.Sequential
        Convolutional layers with batch norm, ReLU, and flatten.
    fc : nn.Sequential
        Fully connected layers reducing CNN output to
        the desired feature dimension.
    """

    def __init__(self, observation_space, vision_params, features_dim=96):
        """
        Initialize the VisionProcessor.

        Parameters
        ----------
        observation_space : tuple
            Shape of the input images (channels, height, width).
        features_dim : int, optional
            Dimensionality of the output feature vector.
            Default: 96.
        """
        super(VisionProcessor, self).__init__(observation_space, features_dim)

        # Set parameters for modularity and flexibility
        self.num_conv_layers = vision_params.get(
            "num_conv_layers", 4
        )  # Number of convolutional layers
        self.conv_channels = vision_params.get(
            "conv_channels", [64, 128, 256, 256]
        )  # Number of output channels for each layer
        self.fc_dims = vision_params.get(
            "fc_dims", [512]
        )  # Dimensions of the fully connected layers

        # Extract dimensions from the observation space
        channels, height, width = observation_space

        # Build the convolutional layers
        conv_layers = []
        in_channels = channels
        for out_channels in self.conv_channels[: self.num_conv_layers]:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(2, 2))
            in_channels = out_channels

        # Adaptive pooling guarantees fixed spatial output regardless of input size
        conv_layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        conv_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*conv_layers)

        # Fixed output size: last_channels * 4 * 4
        last_channels = self.conv_channels[min(self.num_conv_layers, len(self.conv_channels)) - 1]
        total_conv_size = last_channels * 4 * 4

        # Build the fully connected layers
        fc_layers = []
        in_dim = total_conv_size
        for out_dim in self.fc_dims:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU())
            in_dim = out_dim

        # The final layer reduces the dimension to the desired features_dim
        fc_layers.append(nn.Linear(in_dim, features_dim))
        self.fc = nn.Sequential(*fc_layers)

    def forward(self, observations):
        """
        Forward pass through the VisionProcessor.

        Parameters
        ----------
        observations : torch.Tensor
            Batch of images with shape
            (batch_size, channels, height, width).

        Returns
        -------
        torch.Tensor
            Output feature vector of size
            (batch_size, features_dim).
        """
        # Pass the input through the convolutional layers
        x = self.cnn(observations)
        # Pass the output through the fully connected layers
        x = self.fc(x)
        return x


class GraphProcessor(nn.Module):
    """
    Neural network module for processing graph-structured data.

    Uses Graph Attention Network (GAT) layers to capture node
    relationships, then applies fully connected layers to produce
    a fixed-size feature vector.

    Attributes
    ----------
    num_gat_layers : int
        Number of GAT layers (default: 2).
    gat_heads : list of int
        Attention heads per GAT layer.
        Default: [4, 1].
    fc_dims : list of int
        Sizes of fully connected layers after GAT.
        Default: [192].
    gat : nn.ModuleList
        Stacked GAT layers.
    fc : nn.Sequential
        Fully connected layers reducing GAT output to
        the desired feature dimension.
    """

    def __init__(
        self,
        num_graph_node_features,
        graph_params,
        output_dim=96,
        gat_hidden_dim=48,
        num_edge_features=0,
    ):
        """
        Initialize the GraphProcessor.

        Parameters
        ----------
        num_graph_node_features : int
            Number of features per graph node.
        output_dim : int, optional
            Dimensionality of the output feature vector.
            Default: 96.
        num_edge_features : int, optional
            Number of features per edge (passed as edge_dim to GATConv).
            Default: 0 (no edge features).
        """
        super(GraphProcessor, self).__init__()

        # Set parameters for modularity and flexibility
        self.num_gat_layers = graph_params.get("num_gat_layers", 3)  # Number of GAT layers
        self.gat_heads = graph_params.get(
            "gat_heads", [4, 2, 2]
        )  # Number of attention heads for each layer
        self.fc_dims = graph_params.get(
            "fc_dims", [192]
        )  # Dimensions of the fully connected layers
        self.gat_hidden_dim = gat_hidden_dim

        # Build the GAT layers
        gat_layers = []
        in_channels = num_graph_node_features
        edge_dim = num_edge_features if num_edge_features > 0 else None
        for i in range(self.num_gat_layers):
            out_channels = self.gat_hidden_dim * self.gat_heads[i]
            gat_layers.append(
                GATConv(
                    in_channels,
                    self.gat_hidden_dim,
                    heads=self.gat_heads[i],
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

    def forward(self, x, edge_index, batch, edge_attr=None):

        for _i, gat_layer in enumerate(self.gat):
            x = F.relu(gat_layer(x, edge_index, edge_attr=edge_attr))

        x = global_mean_pool(x, batch)
        # print(f"After global_mean_pool: x shape = {x.shape}")

        x = self.fc(x)
        # print(f"Final output shape: {x.shape}")
        return x


class AgentModel(BaseFeaturesExtractor):
    """
    Neural network module for processing both visual and
    graph-structured data.

    Combines outputs of a VisionProcessor and a GraphProcessor
    to produce a unified feature vector for RL tasks.

    Attributes
    ----------
    vision_params : dict
        Config for VisionProcessor (conv layers, channels, FC).
    graph_params : dict
        Config for GraphProcessor (GAT layers, heads, FC).
    fc_dims : list of int
        Sizes of FC layers combining vision and graph outputs.
    dropout_p : float
        Dropout probability (default: 0.25).
    vision_processor : VisionProcessor
        Processes visual input data.
    graph_processor : GraphProcessor
        Processes graph-structured data.
    fc : nn.Sequential
        FC layers combining vision and graph features.
    dropout : nn.Dropout
        Dropout layer to prevent overfitting.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Dict,
        features_dim: int = 192,
        model_config: AgentModelConfig | None = None,
    ):
        """
        Initialize the AgentModel.

        Parameters
        ----------
        observation_space : gym.spaces.Dict
            Dict space with 'vision' and 'node_features' keys.
        features_dim : int, optional
            Dimensionality of the final output feature vector.
            Default: 192.
        """
        super().__init__(observation_space, features_dim=features_dim)

        # Load config defaults (or accept an external config)
        if model_config is None:
            model_config = AgentModelConfig()

        self.disable_vision = model_config.disable_vision
        self.disable_graph = model_config.disable_graph

        # Parameters for modularity and flexibility
        self.vision_params = model_config.to_vision_params()
        self.graph_params = model_config.to_graph_params()

        # Calculate the size of the first fully connected layer
        first_fc_dim = self.vision_params["fc_dims"][-1] + self.graph_params["fc_dims"][-1]

        # Set up the fully connected layers dimensions
        self.fc_dims = [first_fc_dim, first_fc_dim, first_fc_dim // 2, features_dim]

        # Dropout probability
        self.dropout_p = model_config.dropout

        # Initialize VisionProcessor and GraphProcessor with parameters
        vision_shape = observation_space.spaces["vision"].shape
        num_node_features = observation_space.spaces["node_features"].shape[1]
        num_edge_features = observation_space.spaces["edge_attr"].shape[1]

        self.vision_processor = VisionProcessor(
            vision_shape, vision_params=self.vision_params, features_dim=features_dim
        )
        self.graph_processor = GraphProcessor(
            num_node_features,
            graph_params=self.graph_params,
            output_dim=features_dim,
            gat_hidden_dim=model_config.gat_hidden_dim,
            num_edge_features=num_edge_features,
        )

        # Combine the output sizes from both processors
        combined_input_size = (
            self.vision_processor.fc[-1].out_features + self.graph_processor.fc[-1].out_features
        )

        # Define the fully connected layers based on the calculated dimensions
        fc_layers = []
        in_dim = combined_input_size
        for out_dim in self.fc_dims:
            fc_layers.append(nn.Linear(in_dim, out_dim))
            fc_layers.append(nn.ReLU())
            in_dim = out_dim

        # The final fully connected layer
        fc_layers.append(nn.Linear(in_dim, features_dim))
        self.fc = nn.Sequential(*fc_layers)
        self.dropout = nn.Dropout(p=self.dropout_p)

        # Initialize the weights
        self._initialize_weights()

    def forward(self, observations):
        """
        Forward pass through the AgentModel.

        Parameters
        ----------
        observations : dict
            Dict with 'vision' and 'node_features' keys.
            'vision' is a batch of images; 'node_features'
            is the node feature matrix, along with
            'edge_index' and optionally 'edge_attr'/'batch'.

        Returns
        -------
        torch.Tensor
            Output feature vector of size
            (batch_size, features_dim).
        """
        # Process the visual input through the VisionProcessor
        vision_features = self.vision_processor(observations["vision"])
        if self.disable_vision:
            vision_features = torch.zeros_like(vision_features)

        # Handle batched graph data
        batch_size = observations["node_features"].shape[0]
        num_nodes = observations["node_features"].shape[1]

        # Reshape and process graph features
        x = observations["node_features"].view(batch_size * num_nodes, -1).to(torch_dtype)
        edge_index = observations["edge_index"].long()
        edge_index = edge_index + (
            torch.arange(batch_size, device=edge_index.device) * num_nodes
        ).view(-1, 1, 1)
        edge_index = edge_index.view(2, -1)

        # Extract and reshape edge_attr
        num_edges = observations["edge_attr"].shape[1]
        edge_attr = observations["edge_attr"].reshape(batch_size * num_edges, -1).to(torch_dtype)

        batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)

        # Process the graph input through the GraphProcessor
        graph_features = self.graph_processor(x, edge_index, batch, edge_attr=edge_attr)
        if self.disable_graph:
            graph_features = torch.zeros_like(graph_features)

        # Combine vision and graph features
        combined = torch.cat((vision_features, graph_features), dim=1)

        # Apply dropout and fully connected layers
        combined = self.dropout(combined)
        features = self.fc(combined)

        return features

    def _initialize_weights(self):
        """
        Initialize weights using Kaiming normalization.

        Applied to Conv2D and Linear layers. BatchNorm2D
        layers get ones for weights and zeros for biases.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def sanity_check(self, observations):
        with torch.no_grad():
            output = self.forward(observations)
            print(f"Output shape: {output.shape}")
            print(f"Output mean: {output.mean().item():.4f}")
            print(f"Output std: {output.std().item():.4f}")
            vision_mean = self.vision_processor(observations["vision"]).mean().item()
            print(f"Vision features mean: {vision_mean:.4f}")
            graph_data = Data(
                x=observations["node_features"].to(torch_dtype),
                edge_index=observations["edge_index"].long(),
                edge_attr=observations["edge_attr"].to(torch_dtype),
                batch=torch.zeros(
                    observations["node_features"].shape[0],
                    dtype=torch.long,
                ),
            )
            graph_mean = self.graph_processor(graph_data).mean().item()
            print(f"Graph features mean: {graph_mean:.4f}")
