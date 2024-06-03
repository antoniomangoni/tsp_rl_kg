import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class AgentModel(nn.Module):
    def __init__(self, vision_shape, num_graph_features, num_actions, num_nodes, num_edges):
        super(AgentModel, self).__init__()
        
        self.vision_processor = VisionProcessor(vision_shape, out_channels=128)
        self.graph_processor = GraphProcessor(num_graph_features, num_nodes, num_edges, output=128)
        
        combined_input_size = self.vision_processor.output + self.graph_processor.output
        self.fc1 = nn.Linear(combined_input_size, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_actions)
        
        self._initialize_weights()  # Initialize weights

    def forward(self, vision_input, graph_data):
        vision_features = self.vision_processor(vision_input)
        graph_features = self.graph_processor(graph_data)
        
        combined = torch.cat((vision_features, graph_features), dim=1)
        combined = F.relu(self.fc1(combined))
        combined = self.dropout(combined)
        action_probs = self.fc2(combined)
        
        return action_probs

    def _initialize_weights(self):
        # Initialize weights for convolutional and fully connected layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class VisionProcessor(nn.Module):
    def __init__(self, vision_shape, out_channels=128):
        super(VisionProcessor, self).__init__()
        self.output = out_channels
        
        self.conv1 = nn.Conv2d(in_channels=vision_shape[0], out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(64 * vision_shape[1] * vision_shape[2], self.output)
    
    def forward(self, vision_input):
        x = F.relu(self.conv1(vision_input))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc(x))
        return x

class GraphProcessor(nn.Module):
    def __init__(self, num_graph_features, num_nodes, num_edges, output=128):
        super(GraphProcessor, self).__init__()
        self.output = output
        self.conv1 = GATConv(num_graph_features, num_nodes)
        self.conv2 = GATConv(num_nodes, num_edges)
        self.conv3 = GATConv(num_edges, self.output)
    
    def forward(self, graph_data):
        edge_index = graph_data.edge_index
        node_features = graph_data.x
        
        # Handle masks if present
        if graph_data.edge_attr is not None:
            edge_mask = graph_data.edge_attr[:, -1] == 1  # The last feature in edge_attr is the mask
            edge_index = edge_index[:, edge_mask]

        if graph_data.x is not None:
            node_mask = graph_data.x[:, -1] == 1  # The last feature in node_features is the mask
            node_features = node_features[node_mask]
        
        x = F.relu(self.conv1(node_features, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch=graph_data.batch)  # Ensure batch processing
        
        return x
