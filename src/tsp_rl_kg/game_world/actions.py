from enum import IntEnum


class ActionType(IntEnum):
    """All discrete actions available to the agent.

    Movement actions (0-3) map to (dx, dy) offsets.
    Special actions (4-10) trigger abilities or resource collection.
    Values match the legacy integer action space used by stable-baselines3.
    """

    MOVE_LEFT = 0
    MOVE_RIGHT = 1
    MOVE_UP = 2
    MOVE_DOWN = 3
    SCOUT = 4
    BUILD_PATH = 5
    PLACE_ROCK = 6
    COLLECT_UP = 7
    COLLECT_DOWN = 8
    COLLECT_RIGHT = 9
    COLLECT_LEFT = 10


MOVEMENT_DELTAS: dict[ActionType, tuple[int, int]] = {
    ActionType.MOVE_LEFT: (-1, 0),
    ActionType.MOVE_RIGHT: (1, 0),
    ActionType.MOVE_UP: (0, 1),
    ActionType.MOVE_DOWN: (0, -1),
}

COLLECT_DELTAS: dict[ActionType, tuple[int, int]] = {
    ActionType.COLLECT_UP: (0, 1),
    ActionType.COLLECT_DOWN: (0, -1),
    ActionType.COLLECT_RIGHT: (1, 0),
    ActionType.COLLECT_LEFT: (-1, 0),
}
