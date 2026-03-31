import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.utils import to_networkx

from tsp_rl_kg.graph.constitution import DefaultGridConstitution, GraphConstitution
from tsp_rl_kg.graph.feature_encoder import EDGE_PLAYER_TERRAIN, FeatureEncoder, RawIntEncoder
from tsp_rl_kg.graph.projection import CompletenessProjection, ProjectionPolicy

# ---------------------------------------------------------------------------
# State ownership
# ---------------------------------------------------------------------------
# Environment is the single source of truth for terrain, entity, and player
# state.  KnowledgeGraph holds *shared references* to the numpy arrays
# (terrain_index_grid, entity_index_grid) — it may READ them but must NEVER
# WRITE to them.  All mutations to the world go through Environment; KG
# derives node features from the current array values on demand.
#
# Owned by KG: graph tensors, graph_manager, distance
# Shared refs (read-only): terrain_array, entity_array, discovered_grid
# Derived:     player_pos (property → environment.player.grid_x/grid_y)
# ---------------------------------------------------------------------------


class KnowledgeGraph:
    def __init__(
        self,
        environment,
        vision_range,
        completion=1.0,
        plot=False,
        feature_encoder: FeatureEncoder | None = None,
        projection: ProjectionPolicy | None = None,
        constitution: GraphConstitution | None = None,
    ):
        self.environment = environment
        self.terrain_array = environment.terrain_index_grid
        self.entity_array = environment.entity_index_grid
        self.feature_encoder = feature_encoder if feature_encoder is not None else RawIntEncoder()

        assert (
            max(v for idx, v in np.ndenumerate(self.entity_array) if idx != self.player_pos) < 7
        ), "Entity type exceeds the maximum value of 6"

        self.vision_range = vision_range
        if projection is not None:
            self.projection = projection
        else:
            self.projection = CompletenessProjection(
                completion, vision_range, self.terrain_array.shape[0]
            )
        self.distance = self.projection.distance
        # Discovery state lives on Environment; initialise from here since KG knows the distance.
        # When distance is None (e.g. FullGraphProjection), discover the entire grid.
        discovery_radius = (
            self.distance if self.distance is not None else self.terrain_array.shape[0]
        )
        self.environment.init_discovered_area(self.player_pos, discovery_radius)

        self.terrain_z_level = 0
        self.entity_z_level = 1
        self.player_z_level = 2

        # Delegate graph construction to constitution
        if constitution is None:
            constitution = DefaultGridConstitution()
        self.constitution = constitution
        self.graph, self.graph_manager = self.constitution.build(
            environment, self.player_pos, environment.discovered_grid, self.feature_encoder
        )
        self.num_possible_nodes = self.graph.num_nodes
        self.num_possible_edges = self.graph.num_edges

    def _get_embedding(self, z_level, x, y):
        """Derive the embedding for a node from the current array values."""
        if z_level == self.terrain_z_level:
            raw = int(self.terrain_array[x, y])
            return self.feature_encoder.encode_terrain(raw)
        elif z_level == self.entity_z_level:
            # Exclude the player entity from entity nodes
            if (x, y) == self.player_pos:
                raw = 0
            else:
                raw = int(self.entity_array[x, y])
            return self.feature_encoder.encode_entity(raw)
        elif z_level == self.player_z_level:
            return self.feature_encoder.encode_player()
        else:
            raise ValueError(f"Invalid z-level: {z_level}")

    def set_new_node_type(self, idx, new_type):
        if isinstance(new_type, torch.Tensor):
            self.graph.x[idx] = new_type
        else:
            self.graph.x[idx] = torch.tensor(new_type, dtype=torch.float)

    def build_path_node(self, x, y):
        node_idx = self.graph_manager.get_node_idx((x, y), self.entity_z_level)
        new_type = self._get_embedding(self.entity_z_level, x, y)
        self.set_new_node_type(node_idx, new_type)

    def elevate_terrain_node(self, x, y):
        # Environment has already updated terrain_array; just re-derive the node feature.
        node_idx = self.graph_manager.get_node_idx((x, y), self.terrain_z_level)
        new_type = self._get_embedding(self.terrain_z_level, x, y)
        self.set_new_node_type(node_idx, new_type)

    def remove_entity_node(self, x, y):
        # Environment has already set entity_array[x, y] = 0; just re-derive.
        node_idx = self.graph_manager.get_node_idx((x, y), self.entity_z_level)
        new_type = self._get_embedding(self.entity_z_level, x, y)
        self.set_new_node_type(node_idx, new_type)

    @property
    def player_pos(self):
        return (self.environment.player.grid_x, self.environment.player.grid_y)

    def move_player_node(self, x, y):
        self.environment.discover_coordinate(x, y)
        # Update player node position features (only when features have x/y columns)
        player_idx = self.graph_manager.player_idx
        if self.graph.x.shape[1] > 1:
            self.graph.x[player_idx][0] = x
            self.graph.x[player_idx][1] = y
        # Re-wire the single player→terrain edge to the new terrain node
        new_terrain_idx = self.graph_manager.get_node_idx((x, y), self.terrain_z_level)
        d_idx = self.graph_manager.player_edge_direct_idx
        r_idx = self.graph_manager.player_edge_reverse_idx
        self.graph.edge_index[:, d_idx] = torch.tensor(
            [player_idx, new_terrain_idx], dtype=torch.int
        )
        self.graph.edge_index[:, r_idx] = torch.tensor(
            [new_terrain_idx, player_idx], dtype=torch.int
        )
        # Distance is 0 (player is on this terrain node)
        attr = self.feature_encoder.encode_edge(0, EDGE_PLAYER_TERRAIN)
        self.graph.edge_attr[d_idx] = attr
        self.graph.edge_attr[r_idx] = attr
        # Update bookkeeping in graph_manager
        self.graph_manager.rewire_player_edge(new_terrain_idx)

    # Construction methods (create_node, add_nodes, create_edge, add_edge_to_graph,
    # create_terrain_edges, add_entity_edges, compute_total_possible_edges,
    # init_graph_tensors, complete_graph, verify_graph_integrity) moved to
    # DefaultGridConstitution.build().

    # get_graph_distance() — removed; distance computation now lives in
    # CompletenessProjection / ProjectionPolicy.

    def get_cartesian_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def get_manhattan_neighbours(self, coor):
        x, y = coor
        neighbours = []
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            new_x, new_y = x + dx, y + dy
            if self.environment.within_bounds(new_x, new_y):
                neighbours.append((new_x, new_y))
        return neighbours

    def visualise_graph(self, node_size=100, edge_color="tab:gray", show_ticks=True):
        # self.check_path_nodes()
        # Convert to undirected graph for visualization
        G = to_networkx(self.graph, to_undirected=True)

        # Use a 2D spring layout, as z-coordinates are manually assigned
        pos = {}  # nx.spring_layout(G, seed=42)  # 2D layout
        node_xyz = []
        node_colors = []
        for node in sorted(G):
            node_data = self.graph.x[node]
            x, y, z, type_id, mask = node_data
            if mask:
                pos[node] = (x.item(), y.item(), z.item())  # Ensure all items are floats
                color = self.resolve_color(type_id.item(), z.item(), mask.item())
                if color:  # Only append if color is resolved
                    node_xyz.append(pos[node])
                    node_colors.append(color)

        if node_xyz:  # Ensure there are nodes to plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            node_xyz = np.array(node_xyz)
            # invert x-axis to match the game world
            ax.invert_xaxis()
            ax.scatter(*node_xyz.T, s=node_size, color=node_colors, edgecolor="w", depthshade=True)
            # Plot edges
            for edge in G.edges():
                if edge[0] in pos and edge[1] in pos:
                    ax.plot(*np.array([pos[edge[0]], pos[edge[1]]]).T, color=edge_color)

            if show_ticks:
                ax.set_xticks(
                    np.linspace(min(pos[n][0] for n in pos), max(pos[n][0] for n in pos), num=5)
                )
                ax.set_yticks(
                    np.linspace(min(pos[n][1] for n in pos), max(pos[n][1] for n in pos), num=5)
                )
                ax.set_zticks([0, 1])  # Assuming z-levels are either 0 or 1
            else:
                ax.grid(False)
                ax.xaxis.set_ticks([])
                ax.yaxis.set_ticks([])
                ax.zaxis.set_ticks([])

            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Terrain --- Entity --- Agent")
            plt.title("Game World")
            plt.show()

    def resolve_color(self, type_id, z, mask):
        # These colors are in RGB format, normalized to [0, 1]
        # --> green, grey twice, red, brown, black
        entity_colour_map = {
            2: (0.13, 0.33, 0.16),
            3: (0.61, 0.65, 0.62),
            4: (0.61, 0.65, 0.62),
            5: (0.78, 0.16, 0.12),
            6: (0.46, 0.31, 0.04),
        }
        if mask == 0:
            return [0.5, 0.5, 0.5, 0.0]  # transparent grey
        elif z == self.terrain_z_level:
            return [
                c / 255.0 for c in self.environment.terrain_colour_map.get(type_id, (255, 0, 0))
            ]
        elif z == self.entity_z_level:
            return entity_colour_map.get(type_id)
        elif z == self.player_z_level:
            return (0, 0, 0)  # black for player
        return None

    def get_subgraph(self):
        player_terrain_idx = self.graph_manager.get_node_idx(self.player_pos, self.terrain_z_level)
        return self.projection.project(self.graph, self.graph.edge_index, player_terrain_idx)
