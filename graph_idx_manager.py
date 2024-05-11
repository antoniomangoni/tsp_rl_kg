class Graph_Manager:
    def __init__(self):
        self.player_idx = None
        self.node_idx = 0
        self.nodeIdx_id_dict = {}
        self.nodeId_idx_dict = {}
        self.current_edge_idx = 0
        self.nodeTuples_edgeIdx_dict = {}  # Maps edge tuples to indices

    def create_idx(self, pos, z_level):
        node_idx = self.node_idx
        self.nodeIdx_id_dict[node_idx] = (pos, z_level)
        self.nodeId_idx_dict[(pos, z_level)] = node_idx
        self.node_idx += 1
        return node_idx

    def get_node_idx(self, pos, z_level):
        return self.nodeId_idx_dict.get((pos, z_level))
    
    def get_node_pos(self, node_idx):
        return self.nodeIdx_id_dict.get(node_idx)[0]
    
    def create_edge_idx(self, node_idx1, node_idx2):
        if self.current_edge_idx >= self.max_edges:
            print("Max edges reached.")
            return
        direct_edge_idx = self.current_edge_idx
        reverse_edge_idx = self.current_edge_idx + 1
        self.store_edge_indices(node_idx1, node_idx2, direct_edge_idx, reverse_edge_idx)
        self.current_edge_idx += 2
        return direct_edge_idx, reverse_edge_idx

    def store_edge_indices(self, node_idx1, node_idx2, direct_edge_idx, reverse_edge_idx):
        """Store both directions of the edge along with their indices in the graph tensors."""
        if (node_idx1, node_idx2) in self.nodeTuples_edgeIdx_dict:
            return
        if (node_idx2, node_idx1) in self.nodeTuples_edgeIdx_dict:
            return
        if len(self.nodeTuples_edgeIdx_dict) >= self.max_edges:
            print("Max edges reached.")
            return
        self.nodeTuples_edgeIdx_dict[(node_idx1, node_idx2)] = direct_edge_idx
        self.nodeTuples_edgeIdx_dict[(node_idx2, node_idx1)] = reverse_edge_idx 

    def retrieve_edge_indices(self, node_idx1, node_idx2):
        """Retrieve indices for both directions of the edge."""
        direct = self.nodeTuples_edgeIdx_dict.get((node_idx1, node_idx2))
        reverse = self.nodeTuples_edgeIdx_dict.get((node_idx2, node_idx1))
        return direct, reverse
    
    def retrieve_edges_from_node(self, node_idx):
        """Retrieve all edges from a node, including both incoming and outgoing edges."""
        edges = [edge for edge in self.nodeTuples_edgeIdx_dict if node_idx in edge]
        return edges
    
    def set_max_nodes(self, n):
        self.max_nodes = n

    def set_max_edges(self, n):
        self.max_edges = n