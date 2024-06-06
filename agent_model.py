import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class AgentModel(nn.Module):
    def __init__(self, vision_shape, num_graph_features, num_edge_features, num_actions):
        super(AgentModel, self).__init__()
        
        self.vision_processor = VisionProcessor(vision_shape, out_channels=128)
        self.graph_processor = GraphProcessor(num_graph_features, output=128)
        
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
    def __init__(self, num_graph_features, output=128):
        """
        Initializes the GraphProcessor with Graph Attention Convolutional layers.
        
        Parameters:
        - num_graph_features (int): Number of features for each graph node.
        - output (int): Output size of the graph features after processing.
        """
        super(GraphProcessor, self).__init__()
        self.output = output
        self.conv1 = GATConv(num_graph_features, output)
        self.conv2 = GATConv(output, output)
    
    def forward(self, graph_data):
        """
        Forward pass for processing graph data.
        
        Parameters:
        - graph_data: A data object from torch_geometric containing edge_index, x, and optionally edge_attr and batch.
        
        Returns:
        - x: The processed graph features, pooled globally.
        """
        edge_index = graph_data.edge_index
        node_features = graph_data.x
        
        # Validate inputs
        if edge_index is None or node_features is None:
            raise ValueError("edge_index and node_features cannot be None.")
        
        # Handle edge attributes mask if present
        if graph_data.edge_attr is not None:
            edge_mask = graph_data.edge_attr[:, -1] == 1  # Assuming the last feature in edge_attr is the mask
            edge_index = edge_index[:, edge_mask]
        else:
            print("Warning: edge_attr is None, proceeding without edge masks.")
        
        # Handle node features mask if present
        if node_features.shape[1] > 1:  # Check if there are multiple features
            node_mask = node_features[:, -1] == 1  # Assuming the last feature in node_features is the mask
            node_features = node_features[node_mask]
        else:
            print("Warning: node_features do not have a mask, proceeding without node masks.")
        
        # Pass through GATConv layers
        x = F.relu(self.conv1(node_features, edge_index))
        x = self.conv2(x, edge_index)
        
        # Pooling the graph features globally
        x = global_mean_pool(x, batch=graph_data.batch)  # Ensure batch processing
        
        return x