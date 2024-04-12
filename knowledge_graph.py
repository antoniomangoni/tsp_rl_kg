import torch
import torch_geometric as pyg
import networkx as nx
import matplotlib.pyplot as plt

class KnowledgeGraph(pyg.data.Data):
    def __init__(self, environment, agent_vision_range=3, completeness=100):
        super(KnowledgeGraph, self).__init__()
        self.environment = environment
        self.agent_vision_range = agent_vision_range
        self.terrain_features, self.entity_features, self.nodes, self.edges = self.construct_graph()
        self.apply_completeness(completeness)
        self.visualize()

    def node_index(self, x, y, width):
        """Calculate node index based on x and y coordinates."""
        return x * width + y

    def construct_graph(self):
        """Construct the graph from the environment data."""
        height, width = self.environment.heightmap.shape
        terrain_features = []
        entity_features = [[self.environment.player.id]]  # Player's unique ID as its feature
        edge_indices = []

        # Create a node for the player first
        player_node_idx = height * width  # Ensures the player node index is after all terrain nodes

        for x in range(height):
            for y in range(width):
                terrain_object = self.environment.terrain_object_grid[x][y]
                terrain_feature = terrain_object.elevation
                terrain_node_idx = self.node_index(x, y, width)
                terrain_features.append([terrain_feature])

                if terrain_object.entity_on_tile:
                    entity_feature = terrain_object.entity_on_tile.id
                    entity_node_idx = height * width + len(entity_features)
                    entity_features.append([entity_feature])
                    edge_indices.extend([[terrain_node_idx, entity_node_idx], [entity_node_idx, player_node_idx]])

                # Connect nodes in four directions to create the grid
                edge_indices.extend([[terrain_node_idx, self.node_index(nx, ny, width)]
                                     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                     if 0 <= (nx := x + dx) < height and 0 <= (ny := y + dy) < width])

        # Convert to tensors
        terrain_features = torch.tensor(terrain_features, dtype=torch.float)
        entity_features = torch.tensor(entity_features, dtype=torch.float)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Nodes are a concatenation of terrain and entity features, player node added at the end
        nodes = torch.cat((terrain_features, entity_features), dim=0)

        return terrain_features, entity_features, nodes, edge_indices

    def apply_completeness(self, completeness):
        height, width = self.environment.heightmap.shape
        total_terrain_nodes = height * width
        player_node_idx = total_terrain_nodes

        player_x, player_y = self.environment.player.grid_x, self.environment.player.grid_y

        if completeness == 0:
            range_limit = self.agent_vision_range
        elif completeness <= 100:
            range_limit = int((completeness / 100.0) * width) // 2
        else:
            # Extend range by a factor of how much completeness exceeds 100
            extra_range = int(((completeness - 100) / 100.0) * width) // 2
            range_limit = self.agent_vision_range + extra_range

        nodes_to_retain = set()

        # Determine nodes to retain based on the calculated range limit
        for x in range(max(0, player_x - range_limit), min(height, player_x + range_limit + 1)):
            for y in range(max(0, player_y - range_limit), min(width, player_y + range_limit + 1)):
                nodes_to_retain.add(self.node_index(x, y, width))

        # Ensure entities connected to retained terrain nodes are also retained
        retained_nodes = nodes_to_retain.copy()
        for idx in list(nodes_to_retain):
            associated_entity_idx = idx + total_terrain_nodes
            if associated_entity_idx < len(self.nodes):
                retained_nodes.add(associated_entity_idx)

        # Always include the player node
        retained_nodes.add(player_node_idx)

        # Filter nodes and edges
        self.nodes = torch.tensor([self.nodes[i] for i in range(len(self.nodes)) if i in retained_nodes], dtype=torch.float)
        self.edges = torch.tensor([[s, t] for s, t in self.edges.t().tolist() if s in retained_nodes and t in retained_nodes], dtype=torch.long).t().contiguous()
        self.terrain_features = [self.terrain_features[i] for i in nodes_to_retain]
        self.entity_features = [self.entity_features[i - total_terrain_nodes] for i in retained_nodes if i >= total_terrain_nodes]

    def visualize(self):
        """Visualize the knowledge graph."""
        G = nx.Graph()
        num_terrain_nodes = len(self.terrain_features)

        for i, feature in enumerate(self.nodes):
            node_type = 'terrain' if i < num_terrain_nodes else 'entity'
            G.add_node(i, **{node_type: feature.item()})

        G.add_edges_from(self.edges.t().tolist())

        plt.figure(figsize=(8, 6))
        nx.draw(G, with_labels=True, font_weight='bold', node_size=10, node_color='skyblue', font_size=8, font_color='black')
        plt.show()
