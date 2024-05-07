class Graph_Manager:
    def __init__(self):
        self.player_idx = None
        self.node_idx = 0
        self.node_idx_id_dict = {}
        self.node_id_idx_dict = {}
        self.edges = {}  # Maps edge tuples to indices

    def create_idx(self, pos, z_level):
        node_idx = self.node_idx
        self.node_idx_id_dict[node_idx] = (pos, z_level)
        self.node_id_idx_dict[(pos, z_level)] = node_idx
        self.node_idx += 1
        return node_idx

    def get_node_idx(self, pos, z_level):
        return self.node_id_idx_dict.get((pos, z_level))
    
    def get_node_pos(self, node_idx):
        return self.node_idx_id_dict.get(node_idx)[0]

    def add_edges(self, node_idx1, node_idx2, edge_index):
        """Store both directions of the edge along with their indices in the graph tensors."""
        self.edges[(node_idx1, node_idx2)] = edge_index
        self.edges[(node_idx2, node_idx1)] = edge_index + 1  # The next index is the reverse edge

    def get_edge_indices(self, node_idx1, node_idx2):
        """Retrieve indices for both directions of the edge."""
        direct = self.edges.get((node_idx1, node_idx2))
        reverse = self.edges.get((node_idx2, node_idx1))
        return direct, reverse
    