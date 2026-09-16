"""Display-independent policy RGB observations; axes are channel, y, x."""

import numpy as np

# Fixed categorical marker colors, deliberately distinct from terrain colors.
ENTITY_COLOURS = {
    1: (0, 255, 255),
    2: (180, 100, 20),
    3: (220, 180, 100),
    4: (180, 180, 255),
    5: (255, 40, 40),
    6: (255, 220, 0),
}
PLAYER_COLOUR = (255, 0, 255)


def render_semantic_vision(environment, vision_range: int, visible_tiles=None) -> np.ndarray:
    # At least two pixels per tile preserve entity and player markers on tiny maps.
    tile_size = max(2, environment.tile_size)
    side = (2 * vision_range + 1) * tile_size
    image = np.zeros((side, side, 3), dtype=np.uint8)
    px, py = environment.player.grid_x, environment.player.grid_y
    for dx in range(-vision_range, vision_range + 1):
        for dy in range(-vision_range, vision_range + 1):
            x, y = px + dx, py + dy
            if not environment.within_bounds(x, y):
                continue
            if visible_tiles is not None and not visible_tiles[x, y]:
                continue
            sx, sy = (dx + vision_range) * tile_size, (dy + vision_range) * tile_size
            tile = image[sy : sy + tile_size, sx : sx + tile_size]
            tile[:] = environment.terrain_colour_map[int(environment.terrain_index_grid[x, y])]
            entity = int(environment.entity_index_grid[x, y])
            if entity:
                tile[tile_size // 2 :, tile_size // 2 :] = ENTITY_COLOURS[entity]
            if (x, y) == (px, py):
                tile[0, :] = PLAYER_COLOUR
    return image.transpose(2, 0, 1)
