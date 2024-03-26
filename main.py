import seaborn as sns
import matplotlib.pyplot as plt
from time import time

from game_manager import GameManager


if __name__ == '__main__':
    # game_manager = GameManager(map_size=32, tile_size=6)
    # game_manager.run()
    
    number_of_environments = 4000

    # start time
    start_time = time()
    game_managers = []
    for _ in range(number_of_environments):
        game_manager = GameManager(map_size=32, tile_size=6)
        if len(game_manager.environment.outpost_locations) >= 3:
            energy = game_manager.target_manager.target_route_energy
            index = next((i for i, gm in enumerate(game_managers) if energy < gm.target_manager.target_route_energy), len(game_managers))
            game_managers.insert(index, game_manager)

    time_taken = time() - start_time
    print(f'Time taken for {number_of_environments} environments: {time_taken} seconds')
    print(f'Time taken for 1 environment: {time_taken/number_of_environments} seconds')

    # Plot the energy required for the trade route
    energy = [gm.target_manager.target_route_energy for gm in game_managers]
    # sns.histplot(energy, kde=True)
    plt.xlabel('Energy required for trade route')
    plt.ylabel('Frequency')
    plt.title('Distribution of energy required for trade route')
    plt.hist(energy, bins=20)
    plt.show()