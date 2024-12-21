import numpy as np
import matplotlib.pyplot as plt
import heapq

# Load the binary grid from a file
binary_grid = np.load('../static/show/binary_grid.npy')
grid_height, grid_width = binary_grid.shape

# Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Extract obstacle positions from the binary grid
def extract_obstacles(binary_grid):
    obstacles = np.argwhere(binary_grid == 0)
    return set((int(pos[1]), int(pos[0])) for pos in obstacles)

def astar(start, goal, obstacles, search_radius=5):
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): heuristic(start, goal)}

    visited = set()  # Visited nodes
    directions = [
        (1, 0), (-1, 0), (0, 1), (0, -1),  # Cardinal directions
        (1, 1), (-1, -1), (1, -1), (-1, 1)  # Diagonal directions
    ]

    while open_set:
        current = heapq.heappop(open_set)[1]  # Get the node in open set with the lowest f_score

        if current == goal:
            path = []  # Reconstruct path
            while tuple(current) in came_from:
                path.append(current)
                current = came_from[tuple(current)]
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            neighbor_tuple = tuple(neighbor)

            # Check if the neighbor is within bounds and not an obstacle
            if (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height and neighbor_tuple not in obstacles):
                tentative_g_score = g_score[tuple(current)] + (1 if dx == 0 or dy == 0 else 1.414)  # 1.414 for diagonal moves

                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score[neighbor_tuple] = tentative_g_score + heuristic(neighbor, goal)

                    if neighbor_tuple not in visited:
                        heapq.heappush(open_set, (f_score[neighbor_tuple], neighbor_tuple))
                        visited.add(neighbor_tuple)
    return []  # No path is found

# Plotting paths for firefighters
def plot_firefighter_paths(binary_grid, firefighter_paths, firefighter_start, fire_locations):
    fig, ax = plt.subplots()
    ax.imshow(binary_grid, cmap='gray', origin='lower')

    for i, path in enumerate(firefighter_paths):
        print(firefighter_paths)
        path_points = np.array(path)
        ax.plot(path_points[:, 0], path_points[:, 1], '-', alpha=0.7, linewidth=2)

    ax.plot(firefighter_start[0], firefighter_start[1], 'ro', markersize=7, label='Firefighter')
    fire_locations_x = [loc[0] for loc in fire_locations]
    fire_locations_y = [loc[1] for loc in fire_locations]
    ax.plot(fire_locations_x, fire_locations_y, 'rx', markersize=7, label='Fire Locations')
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax.set_title("Firefighter Paths")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.tight_layout()
    plt.savefig('../static/show/shortest_path_fire.png')

def fire_path_calc(num_firefighters, fire_locations):
    # Firefighter start point and fire location
    firefighter_start = (0, 350)
    print(fire_locations)
    #fire_locations = [(500, 600), (920, 220), (1100, 370)]

    obstacles = extract_obstacles(binary_grid)

    # num_firefighters = int(input("Enter the number of firefighters: "))

    # Fire locations assignment
    fire_loc_assigned = [0] * len(fire_locations)  # 0 = unassigned, 1 = assigned

    # Assign firefighters to fire locations
    firefighter_paths = [[] for _ in range(num_firefighters)]
    firefighter_positions = [firefighter_start] * num_firefighters  # Current positions of firefighters

    # Assign each firefighter to the nearest unassigned fire location
    for firefighter in range(num_firefighters):
        distances = [(i, heuristic(firefighter_positions[firefighter], loc))
                    for i, loc in enumerate(fire_locations) if fire_loc_assigned[i] == 0]
        if not distances:
            break
        nearest_loc_index, _ = min(distances, key=lambda x: x[1])
        print(f"Firefighter {firefighter + 1} assigned to fire location {fire_locations[nearest_loc_index]}")
        path = astar(firefighter_positions[firefighter], fire_locations[nearest_loc_index], obstacles)
        firefighter_paths[firefighter].extend(path)
        firefighter_positions[firefighter] = fire_locations[nearest_loc_index]
        fire_loc_assigned[nearest_loc_index] = 1

    # Handle unassigned fire locations
    for loc_index, assigned in enumerate(fire_loc_assigned):
        if assigned == 0:  # Fire location unassigned
            # The closest firefighter to this fire location
            distances = [(i, heuristic(firefighter_positions[i], fire_locations[loc_index]))
                        for i in range(num_firefighters)]
            nearest_firefighter, _ = min(distances, key=lambda x: x[1])
            print(f"Fire location {fire_locations[loc_index]} is unassigned. Assigning to firefighter {nearest_firefighter + 1}.")
            path_extension = astar(firefighter_positions[nearest_firefighter], fire_locations[loc_index], obstacles)
            firefighter_paths[nearest_firefighter].extend(path_extension)
            firefighter_positions[nearest_firefighter] = fire_locations[loc_index]
            fire_loc_assigned[loc_index] = 1
            print(f"Firefighter {nearest_firefighter + 1} extended path: {path_extension}")

    # Print final paths for each firefighter
    for i, path in enumerate(firefighter_paths):
        print(f"Firefighter {i + 1} path: {path}")

    # Plot the paths
    plot_firefighter_paths(binary_grid, firefighter_paths, firefighter_start, fire_locations)