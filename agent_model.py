import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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
    def __init__(self, num_graph_features, num_edge_features, output=128):
        super(GraphProcessor, self).__init__()
        self.output = output
        self.conv1 = GATConv(num_graph_features, output // 2, heads=2, concat=True)
        self.conv2 = GATConv(output, output // 2, heads=2, concat=True)

    def forward(self, graph_data):
        edge_index = graph_data.edge_index
        node_features = graph_data.x

        # Filtering edges based on the last attribute
        if graph_data.edge_attr is not None:
            edge_mask = graph_data.edge_attr[:, -1] == 1
            edge_index = edge_index[:, edge_mask]
        else:
            print("Warning: edge_attr is None, proceeding without edge masks.")

        # Filtering nodes based on the last attribute
        if node_features.shape[1] > 1:
            node_mask = node_features[:, -1] == 1
            node_features = node_features[node_mask]
        else:
            print("Warning: node_features do not have a mask, proceeding without node masks.")

        x = F.relu(self.conv1(node_features, edge_index))
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch=graph_data.batch)

        return x


class AgentModel(nn.Module):
    def __init__(self, observation_space, num_graph_features, num_edge_features, num_actions):
        super(AgentModel, self).__init__()
        # Extract the necessary vision shape from observation space
        vision_shape = observation_space.spaces['vision'].shape
        
        self.vision_processor = VisionProcessor(vision_shape, features_dim=128)
        self.graph_processor = GraphProcessor(num_graph_features, num_edge_features, output=128)
        
        combined_input_size = self.vision_processor.features_dim + self.graph_processor.output
        self.fc1 = nn.Linear(combined_input_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_actions)
        
        self._initialize_weights()

    def forward(self, vision_input, graph_data):
        vision_features = self.vision_processor(vision_input)
        graph_features = self.graph_processor(graph_data)
        
        combined = torch.cat((vision_features, graph_features), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        action_probs = self.fc2(combined)
        
        return action_probs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
