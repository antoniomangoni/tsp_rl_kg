import torch
torch.set_default_dtype(torch.float32)
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

torch_dtype = torch.float32

class VisionProcessor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=96):  # Increased from 64
        super(VisionProcessor, self).__init__(observation_space, features_dim)
        channels, height, width = observation_space
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Increased from 16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # Increased from 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # Increased from 64
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # Increased from 128
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten()
        )
        total_conv_size = 256 * height * width
        self.fc = nn.Sequential(
            nn.Linear(total_conv_size, 512),  # Increased from 256
            nn.ReLU(),
            nn.Linear(512, features_dim)
        )

    def forward(self, observations):
        x = self.cnn(observations)
        x = self.fc(x)
        return x

class GraphProcessor(nn.Module):
    def __init__(self, num_graph_node_features, output_dim=96):  # Increased from 64
        super(GraphProcessor, self).__init__()
        self.output_dim = output_dim
        self.gat1 = GATConv(num_graph_node_features, 48, heads=4)  # Increased from 32
        self.gat2 = GATConv(48 * 4, 96)  # Increased from 64
        self.fc = nn.Sequential(
            nn.Linear(96, 192),  # Increased from 128
            nn.ReLU(),
            nn.Linear(192, output_dim)
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.gat1(x, edge_index))
        x = F.relu(self.gat2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.fc(x)
        return x

class AgentModel(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 192):  # Increased from 128
        super().__init__(observation_space, features_dim=features_dim)
        
        vision_shape = observation_space.spaces['vision'].shape
        num_node_features = observation_space.spaces['node_features'].shape[1]
        
        self.vision_processor = VisionProcessor(vision_shape, features_dim=192)  # Increased from 128
        self.graph_processor = GraphProcessor(num_node_features, output_dim=192)  # Increased from 128
        
        combined_input_size = self.vision_processor.features_dim + self.graph_processor.output_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_input_size, 384),  # Increased from 256
            nn.ReLU(),
            nn.Linear(384, 192),  # Increased from 128
            nn.ReLU(),
            nn.Linear(192, features_dim)
        )
        self.dropout = nn.Dropout(p=0.25)  # Slightly increased from 0.2
        
        self._initialize_weights()

    def forward(self, observations):
        vision_features = self.vision_processor(observations['vision'])
        
        # Handle batched graph data
        batch_size = observations['node_features'].shape[0]
        num_nodes = observations['node_features'].shape[1]
        
        x = observations['node_features'].view(batch_size * num_nodes, -1).to(torch_dtype)
        edge_index = observations['edge_index'].long()
        edge_index = edge_index + (torch.arange(batch_size, device=edge_index.device) * num_nodes).view(-1, 1, 1)
        edge_index = edge_index.view(2, -1)
        
        batch = torch.arange(batch_size, device=x.device).repeat_interleave(num_nodes)
        
        graph_features = self.graph_processor(x, edge_index, batch)
        
        combined = torch.cat((vision_features, graph_features), dim=1)
        combined = self.dropout(combined)
        features = self.fc(combined)
        
        return features

    def _initialize_weights(self):
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