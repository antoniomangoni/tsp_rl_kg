class IDX_Manager:
    def __init__(self):
        self.player_idx = None
        self.idx = 0
        self.idx_id_dict = {}
        self.id_idx_dict = {}
    
    def create_idx(self, pos, z_level):
        # print(f"[create_idx()] Creating index {self.idx} for position {pos} and z_level {z_level}")
        self.idx_id_dict[self.idx] = (pos, z_level)
        self.id_idx_dict[(pos, z_level)] = self.idx
        self.idx += 1
        return self.idx - 1
    
    def remove_idx(self, pos, z_level):
        # print(f"[remove_idx()] Removing index for position {pos} and z_level {z_level}")
        idx = self.id_idx_dict[(pos, z_level)]
        del self.idx_id_dict[idx]
        del self.id_idx_dict[(pos, z_level)]

    def get_idx(self, pos, z_level):
        if self.verify_node_exists(pos, z_level):
            return self.id_idx_dict[(pos, z_level)]
        else:
            raise ValueError(f"Node at position {pos} and z_level {z_level} does not exist in the graph")
    
    def get_pos(self, idx):
        return self.idx_id_dict[idx][0]
    
    def get_type(self, idx):
        return self.idx_id_dict[idx][1]        
    
    def verify_node_exists(self, pos, z_level):
        # print(f"[verify_node_exists()] Verifying node at position {pos} and z_level {z_level}")
        return (pos, z_level) in self.id_idx_dict
