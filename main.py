from game_manager import GameManager

if __name__ == '__main__':
    game_manager = GameManager(map_size=32, tile_size=6)
    exit()
    game_manager.run()
    
    number_of_environments = 100

    # game_managers = [GameManager(map_size=32, tile_size=6) for _ in range(number_of_environments)]
    # for game_manager in game_managers:
    #     if len(game_manager.environment.outpost_locations) < 3:
    #         # remove the game_manager object from the list
    #         game_managers.remove(game_manager)
    #         print("Removed game_manager object from the list.")
    
    # if number_of_environments != len(game_managers):
    #     print(f"Number of environments: {number_of_environments}")

