import numpy as np
from typing import List, Tuple

class RiverPathFinder:

    def __init__(self, heightmap: np.ndarray):
        self.heightmap = heightmap
        self.current_point: Tuple[int, int] = np.unravel_index(np.argmax(self.heightmap), self.heightmap.shape)
        self.river_path: np.ndarray = []

    def find_river_path_with_adam(self, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8) -> np.ndarray:
        """
        Using the Adam optimization algorithm to generate a river path. Momentum is used to prevent the river 
        from getting stuck in local minima.
        """
        m = np.array([0.0, 0.0])
        v = np.array([0.0, 0.0])
        t = 0  

        while True:
            t += 1
            self.river_path.append(self.current_point)
            
            gradient = self._compute_gradient(self.current_point)
            
            m = beta1 * m + (1 - beta1) * gradient
            v = beta2 * v + (1 - beta2) * gradient**2
            
            m_hat = m / (1 - beta1**t)
            v_hat = v / (1 - beta2**t)
            
            move = m_hat / (np.sqrt(v_hat) + epsilon)
            
            self.current_point = self._update_position(self.current_point, move)
            
            if self._termination_condition_met(t):
                break

        return np.array(self.river_path) 

    def _compute_gradient(self, current_point: Tuple[int, int]) -> np.ndarray:
        """
        Compute the gradient at the current point using finite differences.
        """
        x, y = current_point
        gradient_x = (self.heightmap[x + 1, y] - self.heightmap[x - 1, y]) / 2 if 0 < x < self.heightmap.shape[0] - 1 else 0
        gradient_y = (self.heightmap[x, y + 1] - self.heightmap[x, y - 1]) / 2 if 0 < y < self.heightmap.shape[1] - 1 else 0
        return np.array([gradient_x, gradient_y])

    def _update_position(self, current_point: Tuple[int, int], move: np.ndarray) -> Tuple[int, int]:
        """
        Update the position based on the move vector while ensuring the new position is within the heightmap bounds.
        """
        new_point = current_point + move
        new_point = np.clip(new_point, (0, 0), (self.heightmap.shape[0] - 1, self.heightmap.shape[1] - 1))
        return tuple(new_point.astype(int))

    def _termination_condition_met(self, t: int, max_iterations: int = 1000) -> bool:
        """
        Terminate the river path after a certain number of iterations to prevent infinite loops.
        """
        return t >= max_iterations

    def _get_neighbors(self, current_point: Tuple[int, int], size: int = 1) -> List[Tuple[int, int]]:
        """
        Return the neighbors of the current point within the specified size range.
        """
        x, y = current_point
        neighbors = [(x+i, y+j) for i in range(-size, size+1) for j in range(-size, size+1) if (i, j) != (0, 0)]
        neighbors = [(np.clip(x, 0, self.heightmap.shape[0]-1), np.clip(y, 0, self.heightmap.shape[1]-1)) for x, y in neighbors]
        return neighbors

    def _lowest_neighbor(self, current_point: Tuple[int, int], size: int = 1) -> Tuple[int, int]:
        """
        Return the lowest neighbor of the current point within the specified size range.
        """
        neighbors = self._get_neighbors(current_point, size)
        heights = [self.heightmap[x, y] for x, y in neighbors]
        lowest_neighbor_idx = np.argmin(heights)
        return neighbors[lowest_neighbor_idx]
    
    def mark_river_path(self):
        """
        Marks the cells in the river path as 'strong_water' on the heightmap.
        """
        print("river path:", self.river_path)
        for x, y in self.river_path:
            self.heightmap[x, y] = 0
