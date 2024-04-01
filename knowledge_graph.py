import torch
import torch_geometric as pyg

class KnowledgeGraph(pyg.data.Data):
    def __init__(self, node_features, edge_indices, edge_attributes):
        """
        Initialize the Knowledge Graph.
        
        :param node_features: Features of each node in the graph, as a tensor.
        :param edge_indices: Indices of the source and target nodes for each edge, as a tensor.
        :param edge_attributes: Attributes of each edge in the graph, as a tensor.
        """
        super().__init__()
        self.x = node_features
        self.edge_index = edge_indices
        self.edge_attr = edge_attributes

    def __inc__(self, key, value, *args, **kwargs):
        """
        Helper function to correctly increment the index for 'edge_index' when adding nodes.
        """
        if key == 'edge_index':
            return self.x.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    def add_node(self, node_feature):
        """
        Add a new node to the graph.

        :param node_feature: Feature vector of the new node.
        """
        self.x = torch.cat((self.x, node_feature.unsqueeze(0)), 0)

    def add_edge(self, source_index, target_index, edge_attribute):
        """
        Add a new edge to the graph.

        :param source_index: Index of the source node.
        :param target_index: Index of the target node.
        :param edge_attribute: Attribute vector of the new edge.
        """
        new_edge = torch.tensor([[source_index], [target_index]], dtype=torch.long)
        self.edge_index = torch.cat((self.edge_index, new_edge), 1)
        self.edge_attr = torch.cat((self.edge_attr, edge_attribute.unsqueeze(0)), 0)

    def remove_node(self, node_index):
        """
        Remove a node from the graph.

        :param node_index: Index of the node to remove.
        """
        self.x = torch.cat((self.x[:node_index], self.x[node_index+1:]), 0)
        edge_mask = (self.edge_index[0] != node_index) & (self.edge_index[1] != node_index)
        self.edge_index = self.edge_index[:, edge_mask]
        self.edge_attr = self.edge_attr[edge_mask]

    def remove_edge(self, edge_index):
        """
        Remove an edge from the graph.

        :param edge_index: Index of the edge to remove.
        """
        self.edge_index = torch.cat((self.edge_index[:, :edge_index], self.edge_index[:, edge_index+1:]), 1)
        self.edge_attr = torch.cat((self.edge_attr[:edge_index], self.edge_attr[edge_index+1:]), 0)

    def update_node(self, node_index, node_feature):
        """
        Update the attributes of a node.

        :param node_index: Index of the node to update.
        :param node_feature: New feature vector for the node.
        """
        self.x[node_index] = node_feature

    def update_edge(self, edge_index, edge_attribute):
        """
        Update the attributes of an edge.

        :param edge_index: Index of the edge to update.
        :param edge_attribute: New attribute vector for the edge.
        """
        self.edge_attr[edge_index] = edge_attribute

    def get_node(self, node_index):
        """
        Get the attributes of a node.

        :param node_index: Index of the node.
        :return: Feature vector of the node.
        """
        return self.x[node_index]

    def get_edge(self, edge_index):
        """
        Get the attributes of an edge.

        :param edge_index: Index of the edge.
        :return: Attribute vector of the edge.
        """
        return self.edge_attr[edge_index]

    def get_neighbours(self, node_index):
        """
        Get the neighbor nodes of a specific node.

        :param node_index: Index of the node.
        :return: Indices of the neighboring nodes.
        """
        neighbours = self.edge_index[1, self.edge_index[0] == node_index]
        return neighbours
