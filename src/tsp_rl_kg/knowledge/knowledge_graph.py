from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx

from tsp_rl_kg.graph.constitution import DefaultGridConstitution
from tsp_rl_kg.graph.feature_encoder import EDGE_PLAYER_TERRAIN, RawIntEncoder
from tsp_rl_kg.knowledge.state import KnowledgeState


class KnowledgeGraph:
    """Graph features derive exclusively from last-observed knowledge."""

    terrain_z_level = 0
    entity_z_level = 1
    player_z_level = 2

    def __init__(
        self,
        environment,
        vision_range,
        completion=1.0,
        plot=False,
        feature_encoder=None,
        projection=None,
        constitution=None,
        seed=0,
        template=None,
    ):
        self.environment = environment
        self.feature_encoder = feature_encoder or RawIntEncoder()
        self.vision_range = vision_range
        self.projection = projection
        self.state = KnowledgeState(environment, completion, seed, template)
        self.visible_tiles = self.state.sense(environment, self.player_pos, vision_range)
        # The constitution may see remembered values only, not live hidden arrays.
        remembered = SimpleNamespace(
            width=environment.width,
            height=environment.height,
            terrain_index_grid=self.state.terrain,
            entity_index_grid=self.state.entities,
        )
        self.constitution = constitution or DefaultGridConstitution()
        self.graph, self.graph_manager = self.constitution.build(
            remembered, self.player_pos, self.state.known_tiles, self.feature_encoder
        )
        self.num_possible_nodes = self.graph.num_nodes
        self.num_possible_edges = self.graph.num_edges

    @property
    def player_pos(self):
        return (self.environment.player.grid_x, self.environment.player.grid_y)

    def _refresh(self, tiles):
        for x, y in np.argwhere(tiles):
            terrain_idx = self.graph_manager.get_node_idx((x, y), self.terrain_z_level)
            entity_idx = self.graph_manager.get_node_idx((x, y), self.entity_z_level)
            self.graph.x[terrain_idx] = self.feature_encoder.encode_terrain(
                int(self.state.terrain[x, y])
            )
            self.graph.x[entity_idx] = self.feature_encoder.encode_entity(
                int(self.state.entities[x, y])
            )

    def sense(self, radius=None):
        radius = self.vision_range if radius is None else radius
        self.visible_tiles = self.state.sense(self.environment, self.player_pos, radius)
        self._refresh(self.visible_tiles)

    def observe_tile(self, x, y):
        tiles = np.zeros_like(self.state.known_tiles)
        tiles[x, y] = True
        self.state.observe(self.environment, tiles)
        self._refresh(tiles)

    def build_path_node(self, x, y):
        self.observe_tile(x, y)

    def elevate_terrain_node(self, x, y):
        self.observe_tile(x, y)

    def remove_entity_node(self, x, y):
        self.observe_tile(x, y)

    def move_player_node(self, x, y):
        gm = self.graph_manager
        self.graph.x[gm.player_idx] = self.feature_encoder.encode_player()
        terrain_idx = gm.get_node_idx((x, y), self.terrain_z_level)
        self.graph.edge_index[:, gm.player_edge_direct_idx] = torch.tensor(
            [gm.player_idx, terrain_idx]
        )
        self.graph.edge_index[:, gm.player_edge_reverse_idx] = torch.tensor(
            [terrain_idx, gm.player_idx]
        )
        attr = self.feature_encoder.encode_edge(0, EDGE_PLAYER_TERRAIN)
        self.graph.edge_attr[gm.player_edge_direct_idx] = attr
        self.graph.edge_attr[gm.player_edge_reverse_idx] = attr
        gm.rewire_player_edge(terrain_idx)
        self.sense()

    def get_subgraph(self):
        selected = [self.graph_manager.player_idx]
        for idx in range(self.graph.num_nodes):
            coords, level = self.graph_manager.nodeIdx_id_dict[idx]
            if level != self.player_z_level and self.state.known_tiles[coords]:
                selected.append(idx)
        selected = torch.tensor(sorted(selected), dtype=torch.long)
        edges, attrs = subgraph(
            selected,
            self.graph.edge_index.long(),
            self.graph.edge_attr,
            relabel_nodes=True,
            num_nodes=self.graph.num_nodes,
        )
        result = Data(
            x=self.graph.x[selected], edge_index=edges, edge_attr=attrs, world_node_ids=selected
        )
        if self.projection is not None:
            world_idx = self.graph_manager.get_node_idx(self.player_pos, self.terrain_z_level)
            local_idx = int((selected == world_idx).nonzero()[0])
            return self.projection.project(result, result.edge_index, local_idx)
        return result

    def visualise_graph(self, node_size=100, edge_color="tab:gray", show_ticks=True):
        graph = self.get_subgraph()
        if not hasattr(graph, "world_node_ids") or graph.world_node_ids is None:
            raise ValueError("Visualization requires world_node_ids metadata")
        positions = {}
        for local_idx, world_idx in enumerate(graph.world_node_ids.tolist()):
            (x, y), level = self.graph_manager.nodeIdx_id_dict[world_idx]
            positions[local_idx] = (x, y, level)
        ax = plt.figure().add_subplot(111, projection="3d")
        points = np.array(list(positions.values()))
        ax.scatter(*points.T, s=node_size)
        for left, right in to_networkx(graph).edges():
            ax.plot(*np.array([positions[left], positions[right]]).T, color=edge_color)
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
        plt.show()
