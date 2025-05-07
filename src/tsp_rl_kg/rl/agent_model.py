import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

torch_dtype = torch.float64

class VisionProcessor(BaseFeaturesExtractor):
    """
    VisionProcessor is a neural network module designed for processing visual input (image data).
    It consists of a configurable number of convolutional layers followed by fully connected layers,
    enabling it to extract hierarchical features from image observations and output a fixed-size feature vector.

    Attributes:
    -----------
    num_conv_layers : int
        The number of convolutional layers in the model. This is set to 4 by default.
    conv_channels : list of int
        A list containing the number of output channels for each convolutional layer. 
        The default is [32, 64, 128, 256], meaning the first layer has 32 channels, the second 64, and so on.
    fc_dims : list of int
        A list containing the sizes of the fully connected layers following the convolutional layers.
        The default is [512], meaning a single fully connected layer with 512 units.
    cnn : nn.Sequential
        A sequential container that holds the convolutional and batch normalization layers,
        along with ReLU activations and a flattening operation to prepare the data for fully connected layers.
    fc : nn.Sequential
        A sequential container that holds the fully connected layers, including ReLU activations,
        which reduce the output of the CNN to the desired feature dimension.

    Methods:
    --------
    __init__(self, observation_space, features_dim=96):
        Initializes the VisionProcessor with the specified observation space and feature dimension.

    forward(self, observations):
        Defines the forward pass of the model, processing input image data through the convolutional layers
        followed by fully connected layers to produce a feature vector.
    """
    
    def __init__(self, observation_space, vision_params, features_dim=96):
        """
        Initializes the VisionProcessor object.

        Parameters:
        -----------
        observation_space : tuple
            The shape of the input images, typically in the form (channels, height, width).
            This defines the input dimensions for the convolutional layers.
        features_dim : int, optional
            The dimensionality of the output feature vector produced by the model. The default value is 96.
        """
        super(VisionProcessor, self).__init__(observation_space, features_dim)

        # Set parameters for modularity and flexibility
        self.num_conv_layers = vision_params.get("num_conv_layers", 4)  # Number of convolutional layers
        self.conv_channels = vision_params.get("conv_channels", [64, 128, 256, 256])  # Number of output channels for each layer
        self.fc_dims = vision_params.get("fc_dims",  [512])  # Dimensions of the fully connected layers

        # Extract dimensions from the observation space
        channels, height, width = observation_space
        
        # Build the convolutional layers
        conv_layers = []
        in_channels = 3
        for out_channels in self.conv_channels[:self.num_conv_layers]:
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels
        
        # Add a flattening layer to the sequence
        conv_layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*conv_layers)
        
        # Calculate the total output size after the convolutional layers
        total_conv_size = self.conv_channels[min(self.num_conv_layers, len(self.conv_channels)) - 1] * height * width
        
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

        Parameters:
        -----------
        observations : torch.Tensor
            A batch of images with shape (batch_size, channels, height, width) to be processed.

        Returns:
        --------
        torch.Tensor
            The output feature vector of size (batch_size, features_dim).
        """
        # Pass the input through the convolutional layers
        x = self.cnn(observations)
        # Pass the output through the fully connected layers
        x = self.fc(x)
        return x


class GraphProcessor(nn.Module):
    """
    GraphProcessor is a neural network module designed for processing graph-structured data.
    It utilizes Graph Attention Network (GAT) layers to capture relationships between nodes in a graph
    and then applies fully connected layers to produce a fixed-size feature vector.

    Attributes:
    -----------
    num_gat_layers : int
        The number of GAT (Graph Attention Network) layers in the model. This is set to 2 by default.
    gat_heads : list of int
        A list containing the number of attention heads for each GAT layer.
        The default is [4, 1], meaning the first GAT layer has 4 heads, and the second has 1 head.
    fc_dims : list of int
        A list containing the sizes of the fully connected layers following the GAT layers.
        The default is [192], meaning a single fully connected layer with 192 units.
    gat : nn.ModuleList
        A module list containing the GAT layers, allowing for flexible stacking of multiple GAT layers.
    fc : nn.Sequential
        A sequential container that holds the fully connected layers, including ReLU activations,
        which reduce the output of the GAT layers to the desired feature dimension.

    Methods:
    --------
    __init__(self, num_graph_node_features, output_dim=96):
        Initializes the GraphProcessor with the specified number of input node features and output dimensions.

    forward(self, x, edge_index, batch):
        Defines the forward pass of the model, processing input node features through the GAT layers,
        followed by global mean pooling and fully connected layers to produce a feature vector.
    """
    
    def __init__(self, num_graph_node_features, graph_params, output_dim=96):
        """
        Initializes the GraphProcessor object.

        Parameters:
        -----------
        num_graph_node_features : int
            The number of features for each node in the graph, which determines the input dimension for the first GAT layer.
        output_dim : int, optional
            The dimensionality of the output feature vector produced by the model. The default value is 96.
        """
        super(GraphProcessor, self).__init__()

        # Set parameters for modularity and flexibility
        self.num_gat_layers = graph_params.get("num_gat_layers", 3)  # Number of GAT layers 
        self.gat_heads = graph_params.get("gat_heads", [4, 2, 2])  # Number of attention heads for each layer
        self.fc_dims = graph_params.get("fc_dims", [192])  # Dimensions of the fully connected layers
        
        # Build the GAT layers
        gat_layers = []
        in_channels = num_graph_node_features
        for i in range(self.num_gat_layers):
            out_channels = 48 * self.gat_heads[i]
            gat_layers.append(GATConv(in_channels, 48, heads=self.gat_heads[i]))
            # print(f"GAT layer {i}: in_channels={in_channels}, out_channels={out_channels}")
            in_channels = out_channels

        self.gat = nn.ModuleList(gat_layers)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, self.fc_dims[0]),
            nn.ReLU(),
            nn.Linear(self.fc_dims[0], output_dim)
        )

    def forward(self, x, edge_index, batch):
        # print(f"Input x shape: {x.shape}")
        # print(f"Input edge_index shape: {edge_index.shape}")
        # print(f"Input batch shape: {batch.shape}")

        for i, gat_layer in enumerate(self.gat):
            # print(f"Before GAT layer {i}: x shape = {x.shape}")
            x = F.relu(gat_layer(x, edge_index))
            # print(f"After GAT layer {i}: x shape = {x.shape}")
        
        x = global_mean_pool(x, batch)
        # print(f"After global_mean_pool: x shape = {x.shape}")
        
        x = self.fc(x)
        # print(f"Final output shape: {x.shape}")
        return x


class AgentModel(BaseFeaturesExtractor):
    """
    AgentModel is a neural network module designed to process both visual and graph-structured data.
    It combines the outputs of a VisionProcessor and a GraphProcessor to produce a unified feature vector
    that can be used in reinforcement learning or other machine learning tasks.

    Attributes:
    -----------
    vision_params : dict
        Parameters for configuring the VisionProcessor, including the number of convolutional layers,
        the number of channels for each layer, and the dimensions of the fully connected layers.
    graph_params : dict
        Parameters for configuring the GraphProcessor, including the number of GAT layers,
        the number of attention heads for each layer, and the dimensions of the fully connected layers.
    fc_dims : list of int
        A list containing the sizes of the fully connected layers that combine the outputs of the VisionProcessor
        and GraphProcessor. This is automatically calculated based on the output sizes of these processors.
    dropout_p : float
        The probability of dropping out units in the dropout layer to prevent overfitting. Default is 0.25.
    vision_processor : VisionProcessor
        An instance of the VisionProcessor class, which processes visual input data.
    graph_processor : GraphProcessor
        An instance of the GraphProcessor class, which processes graph-structured data.
    fc : nn.Sequential
        A sequential container holding the fully connected layers that combine the processed visual and graph features.
    dropout : nn.Dropout
        A dropout layer used to prevent overfitting.

    Methods:
    --------
    __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 192):
        Initializes the AgentModel with the specified observation space and feature dimension.

    forward(self, observations):
        Defines the forward pass of the model, processing visual and graph inputs,
        combining their features, and passing them through fully connected layers.

    _initialize_weights(self):
        Initializes the weights of the convolutional and fully connected layers using Kaiming normalization.
    """
    
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 192):
        """
        Initializes the AgentModel object.

        Parameters:
        -----------
        observation_space : gym.spaces.Dict
            A dictionary space containing the observation spaces for vision and graph data.
            The 'vision' key should map to a space with the shape of the image input,
            and the 'node_features' key should map to a space with the shape of the graph node features.
        features_dim : int, optional
            The dimensionality of the final output feature vector produced by the model. The default value is 192.
        """
        super().__init__(observation_space, features_dim=features_dim)
        
        # Parameters for modularity and flexibility
        self.vision_params = {'num_conv_layers': 4, 'conv_channels': [64, 128, 256, 512], 'fc_dims': [512]}
        self.graph_params = {'num_gat_layers': 3, 'gat_heads': [4, 4, 4], 'fc_dims': [256]}
        
        # Calculate the size of the first fully connected layer
        first_fc_dim = self.vision_params['fc_dims'][-1] + self.graph_params['fc_dims'][-1]
        
        # Set up the fully connected layers dimensions
        self.fc_dims = [first_fc_dim, first_fc_dim, first_fc_dim // 2, features_dim]
        
        # Dropout probability
        self.dropout_p = 0.25
        
        # Initialize VisionProcessor and GraphProcessor with parameters
        vision_shape = observation_space.spaces['vision'].shape
        num_node_features = observation_space.spaces['node_features'].shape[1]
        
        self.vision_processor = VisionProcessor(vision_shape, vision_params=self.vision_params, features_dim=features_dim)
        self.graph_processor = GraphProcessor(num_node_features, graph_params=self.graph_params, output_dim=features_dim)
        
        # Combine the output sizes from both processors
        combined_input_size = self.vision_processor.fc[-1].out_features + self.graph_processor.fc[-1].out_features
        
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

        Parameters:
        -----------
        observations : dict
            A dictionary containing 'vision' and 'node_features' keys. 'vision' should be a batch of images,
            and 'node_features' should be the node feature matrix for the graph, along with 'edge_index' 
            and optionally 'edge_attr' and 'batch'.

        Returns:
        --------
        torch.Tensor
            The output feature vector of size (batch_size, features_dim).
        """
        # Process the visual input through the VisionProcessor
        vision_features = self.vision_processor(observations['vision'])
        
        # Handle batched graph data
        batch_size = observations['node_features'].shape[0]
        num_nodes = observations['node_features'].shape[1]
        
        # Reshape and process graph features
        x = observations['node_features'].view(batch_size * num_nodes, -1).to(torch_dtype)
        edge_index = observations['edge_index'].long()
        edge_index = edge_index + (torch.arange(batch_size, device=edge_index.device) * num_nodes).view(-1, 1, 1)
        edge_index = edge_index.view(2, -1)
        
        batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        
        # Process the graph input through the GraphProcessor
        graph_features = self.graph_processor(x, edge_index, batch)
        
        # Combine vision and graph features
        combined = torch.cat((vision_features, graph_features), dim=1)
        
        # Apply dropout and fully connected layers
        combined = self.dropout(combined)
        features = self.fc(combined)
        
        return features

    def _initialize_weights(self):
        """
        Initializes the weights of the model using Kaiming normalization.
        This ensures better convergence during training by setting the initial weights appropriately.

        Kaiming normalization is applied to Conv2D and Linear layers, while BatchNorm2D layers are initialized
        with ones for weights and zeros for biases.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
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
            print(f"Vision features mean: {self.vision_processor(observations['vision']).mean().item():.4f}")
            print(f"Graph features mean: {self.graph_processor(Data(x=observations['node_features'].to(torch_dtype), edge_index=observations['edge_index'].long(), edge_attr=observations['edge_attr'].to(torch_dtype), batch=torch.zeros(observations['node_features'].shape[0], dtype=torch.long))).mean().item():.4f}")
