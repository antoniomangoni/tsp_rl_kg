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
        entity_features = []
        edge_indices = []

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
                    edge_indices.extend([[terrain_node_idx, entity_node_idx]])

                # Connect nodes in four directions to create the grid
                edge_indices.extend([[terrain_node_idx, self.node_index(nx, ny, width)]
                                     for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                                     if 0 <= (nx := x + dx) < height and 0 <= (ny := y + dy) < width])

        # Create a node for the player
        player_node_idx = height * width  # Ensures the player node index is after all terrain nodes
        entity_features.append([self.environment.player.id])  # Player's unique ID as its feature
        # Connect the player to the terrain node it occupies
        edge_indices.append([player_node_idx, self.node_index(self.environment.player.grid_x, self.environment.player.grid_y, width)])

        # Convert to tensors
        terrain_features = torch.tensor(terrain_features, dtype=torch.float)
        entity_features = torch.tensor(entity_features, dtype=torch.float)
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # Nodes are a concatenation of terrain and entity features, player node added at the end
        nodes = torch.cat((terrain_features, entity_features), dim=0)

        return terrain_features, entity_features, nodes, edge_indices

    def apply_completeness(self, completeness):
        """Adjust the graph's nodes and edges based on the completeness parameter."""
        height, width = self.environment.heightmap.shape
        total_terrain_nodes = height * width

        # Define the player node index
        player_node_idx = total_terrain_nodes

        # Determine which terrain nodes are in the agent's vision
        player_x, player_y = self.environment.player.grid_x, self.environment.player.grid_y
        terrain_nodes_in_vision = set()
        for x in range(player_x - self.agent_vision_range, player_x + self.agent_vision_range + 1):
            for y in range(player_y - self.agent_vision_range, player_y + self.agent_vision_range + 1):
                if 0 <= x < height and 0 <= y < width:
                    terrain_nodes_in_vision.add(self.node_index(x, y, width))

        # Calculate the number of terrain nodes to retain based on the completeness parameter
        num_terrain_nodes_to_retain = int((completeness / 100.0) * total_terrain_nodes)

        # If completeness is not 100, remove terrain nodes and their corresponding entity nodes
        if completeness < 100:
            # Sort nodes by distance to player and retain the closest ones
            all_terrain_nodes = sorted(range(total_terrain_nodes), 
                                       key=lambda idx: abs(idx % width - player_x) + abs(idx // width - player_y))
            terrain_nodes_to_retain = set(all_terrain_nodes[:num_terrain_nodes_to_retain])

            # Include terrain nodes within vision regardless of completeness, to ensure player's surroundings are always included
            terrain_nodes_to_retain.update(terrain_nodes_in_vision)

            # Now, determine entity nodes to retain
            entity_nodes_to_retain = set()
            for entity_node_idx in range(total_terrain_nodes, len(self.nodes)):
                # The corresponding terrain node for each entity is the one before it in the list (due to how they were added)
                corresponding_terrain_node_idx = entity_node_idx - 1
                if corresponding_terrain_node_idx in terrain_nodes_to_retain:
                    entity_nodes_to_retain.add(entity_node_idx)

            # Combine terrain and entity nodes to retain
            nodes_to_retain = terrain_nodes_to_retain.union(entity_nodes_to_retain)

            # Filter nodes based on retained set
            self.nodes = torch.tensor([self.nodes[i] for i in range(len(self.nodes)) if i in nodes_to_retain], dtype=torch.float)
            self.terrain_features = torch.tensor([self.terrain_features[i] for i in range(len(self.terrain_features)) if i in terrain_nodes_to_retain], dtype=torch.float)
            self.entity_features = torch.tensor([self.entity_features[i - total_terrain_nodes] for i in entity_nodes_to_retain], dtype=torch.float)

            # Update edges
            retained_edges = []
            for edge in self.edges.t().tolist():
                if edge[0] in nodes_to_retain and edge[1] in nodes_to_retain:
                    retained_edges.append(edge)
            self.edges = torch.tensor(retained_edges, dtype=torch.long).t().contiguous()

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
