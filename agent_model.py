import torch
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

class MaskedGCN(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super(MaskedGCN, self).__init__()
        self.conv1 = GCNConv(in_features, out_features)

    def forward(self, data):
        x, edge_index, mask = data.x, data.edge_index, data.mask

        # Apply mask: zero out features of inactive nodes
        x = x * mask.unsqueeze(1)  # Assuming mask is [num_nodes,] and features are [num_nodes, num_features]
        
        # Perform the convolution
        x = self.conv1(x, edge_index)

        # Optionally zero out the features again after convolution if needed
        x = x * mask.unsqueeze(1)

        return x

# Example data setup
num_nodes = 100
num_features = 16

# Random features, edges, and mask
features = torch.randn(num_nodes, num_features)
edge_index = torch.randint(0, num_nodes, (2, 300))  # Example edge index for simplicity
mask = torch.randint(0, 2, (num_nodes,))  # Binary mask where 1 indicates active and 0 indicates inactive

# Create the graph data object
data = Data(x=features, edge_index=edge_index)
data.mask = mask  # Add mask to the data object

# Create and use the model
model = MaskedGCN(num_features, 32)
output = model(data)

print(output)
