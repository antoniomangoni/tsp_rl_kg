import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class VisionProcessor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super(VisionProcessor, self).__init__(observation_space, features_dim)
        channels, height, width = observation_space  # Unpack the dimensions directly
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Since padding=1 and kernel_size=3, the output dimensions are the same as input dimensions
        total_conv_size = 64 * height * width  # Compute based on actual dimensions
        self.fc = nn.Linear(total_conv_size, features_dim)

    def forward(self, observations):
        x = self.cnn(observations)
        x = self.fc(x)
        return x


class GraphProcessor(nn.Module):
    def __init__(self, num_graph_node_features, output=128):
        super(GraphProcessor, self).__init__()
        self.output = output
        self.linear1 = nn.Linear(num_graph_node_features, output)
        self.linear2 = nn.Linear(output, output)

    def forward(self, graph_data):
        node_features = graph_data.x
        edge_index = graph_data.edge_index

        # Apply linear layers
        x = F.relu(self.linear1(node_features))
        x = self.linear2(x)
        
        # Global mean pooling over nodes
        x = torch.mean(x, dim=1)

        return x

class AgentModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        
        vision_shape = observation_space.spaces['vision'].shape
        
        self.vision_processor = VisionProcessor(vision_shape, features_dim=128)
        self.graph_processor = GraphProcessor(observation_space.spaces['node_features'].shape[1], output=128)
        
        combined_input_size = self.vision_processor.features_dim + self.graph_processor.output
        self.fc1 = nn.Linear(combined_input_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, features_dim)
        
        self._initialize_weights()

    def forward(self, observations):
        vision_features = self.vision_processor(observations['vision'])
        
        graph_data = Data(
            x=observations['node_features'].float(),
            edge_index=observations['edge_index'].long(),
            edge_attr=observations['edge_attr'].float()
        )
        
        graph_features = self.graph_processor(graph_data)
        
        # Ensure both features have the same batch dimension
        if len(vision_features.shape) == 2:
            vision_features = vision_features.unsqueeze(0)
        if len(graph_features.shape) == 1:
            graph_features = graph_features.unsqueeze(0)
        
        combined = torch.cat((vision_features, graph_features), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        features = self.fc2(combined)
        
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class AgentModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim=features_dim)
        
        vision_shape = observation_space.spaces['vision'].shape
        
        self.vision_processor = VisionProcessor(vision_shape, features_dim=128)
        self.graph_processor = GraphProcessor(observation_space.spaces['node_features'].shape[1], output=128)
        
        combined_input_size = self.vision_processor.features_dim + self.graph_processor.output
        self.fc1 = nn.Linear(combined_input_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, features_dim)
        
        self._initialize_weights()

    def forward(self, observations):
        vision_features = self.vision_processor(observations['vision'])
        
        # Create a PyTorch Geometric Data object from flattened observations
        graph_data = Data(
            x=observations['node_features'].float(),
            edge_index=observations['edge_index'].long(),
            edge_attr=observations['edge_attr'].float()
        )
        
        # Ensure all tensors are on the same device
        device = vision_features.device
        graph_data = graph_data.to(device)
        
        graph_features = self.graph_processor(graph_data)
        
        combined = torch.cat((vision_features, graph_features), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        features = self.fc2(combined)
        
        return features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
