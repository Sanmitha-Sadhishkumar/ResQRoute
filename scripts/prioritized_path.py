#Prioritized
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Load the binary grid 
binary_grid = np.load('../static/show/binary_grid.npy')
grid_height, grid_width = binary_grid.shape

# Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Extract obstacle positions
def extract_obstacles(binary_grid):
    obstacles = np.argwhere(binary_grid == 0)
    return set((int(pos[1]), int(pos[0])) for pos in obstacles)

# Location dangerous check
def is_dangerous(x, y, dangerous_locations, danger_radius=30):
    for dx, dy in dangerous_locations:
        if abs(x - dx) <= danger_radius and abs(y - dy) <= danger_radius:
            return True
    return False

def astar(start, goal, obstacles, dangerous_locations):
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
        current = heapq.heappop(open_set)[1]  # Lowest f_score

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
                # Add penalty for dangerous locations
                penalty = 50 if is_dangerous(neighbor[0], neighbor[1], dangerous_locations) else 0

                tentative_g_score = g_score[tuple(current)] + (1 if dx == 0 or dy == 0 else 1.414) + penalty

                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score[neighbor_tuple] = tentative_g_score + heuristic(neighbor, goal)

                    if neighbor_tuple not in visited:
                        heapq.heappush(open_set, (f_score[neighbor_tuple], neighbor_tuple))
                        visited.add(neighbor_tuple)
    return []  # No path is found


# Sort occupant locations with priority
def sort_occupant_locations(entry_point, child_locations, elder_locations, adult_locations):
    sorted_children = sorted(child_locations, key=lambda loc: heuristic(entry_point, loc), reverse=True)
    sorted_elderly = sorted(elder_locations, key=lambda loc: heuristic(entry_point, loc), reverse=True)
    sorted_adults = sorted(adult_locations, key=lambda loc: heuristic(entry_point, loc), reverse=True)
    return sorted_children + sorted_elderly + sorted_adults

# Plotting paths
def plot_firefighter_paths(binary_grid, firefighter_paths, fire_locations, occupant_locations, 
                            child_loc, elder_loc, adult_loc):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(binary_grid, cmap='gray', origin='lower')

    for i, path in enumerate(firefighter_paths):
        if not path:
            print(f"Warning: Firefighter {i+1} has no path.")
            continue
        path_points = np.array(path)
       # if path_points.ndim == 1: # Ensure path in 2D
            #path_points = path_points.reshape(-1, 1)

        ax.plot(path_points[:, 0], path_points[:, 1], '-', alpha=0.7, linewidth=2,
                label=f'Firefighter {i+1} Path')
    
    # Plot children
    ax.plot([loc[0] for loc in child_loc],
            [loc[1] for loc in child_loc],
            'go', markersize=7, label='Children Locations')
    
    # Plot elderly
    ax.plot([loc[0] for loc in elder_loc],
            [loc[1] for loc in elder_loc],
            'orange', marker='o', markersize=7, linestyle='None', label='Elderly Locations')
    
    # Plot adults
    ax.plot([loc[0] for loc in adult_loc],
            [loc[1] for loc in adult_loc],
            'purple', marker='o', markersize=7, linestyle='None', label='Adult Locations')

    # Fire locations
    ax.plot([loc[0] for loc in fire_locations],
            [loc[1] for loc in fire_locations],
            'rx', markersize=7, label='Fire Locations')

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
    ax.set_title("Firefighter Rescue Paths")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.tight_layout()
    plt.savefig('../static/show/shortest_path_fire.png')
    
#Locations
def fire_path_calc(num_firefighters, adult_loc, child_loc, elder_loc):
    firefighter_start = (0, 350)
    # child_loc = [(920, 220), (1100, 370)]
    # elder_loc = [(1400, 200), (500, 600)]
    # adult_loc = [(1200, 600)]
    fire_locations = [(1200, 420), (820, 180), (300, 300),(500,500)]

    occupant_locations = sort_occupant_locations(firefighter_start, child_loc, elder_loc, adult_loc)
    obstacles = extract_obstacles(binary_grid)

    # num_firefighters = int(input("Enter the number of firefighters: "))

    # Initialize tracking
    firefighter_paths = [[] for _ in range(num_firefighters)]
    firefighter_positions = [firefighter_start] * num_firefighters
    occupant_loc_assigned = [0] * len(occupant_locations)

    # Assign firefighters
    for firefighter in range(min(num_firefighters, len(occupant_locations))):
        loc_index = firefighter
        print(f"Firefighter {firefighter + 1} assigned to occupant location {occupant_locations[loc_index]}")

        path = astar(firefighter_positions[firefighter],
                    occupant_locations[loc_index],
                    obstacles,
                    fire_locations)

        firefighter_paths[firefighter].extend(path)
        firefighter_positions[firefighter] = occupant_locations[loc_index]
        occupant_loc_assigned[loc_index] = 1

    # Assign remaining firefighters in a priority-based circular manner
    if num_firefighters > len(occupant_locations):
        for firefighter in range(len(occupant_locations), num_firefighters):
            # Circular index for occupant locations, maintaining priority order
            loc_index = firefighter % len(occupant_locations)

            print(f"Firefighter {firefighter + 1} follows the path to {occupant_locations[loc_index]}")

            # If the location is assigned, find the closest unassigned location
            if occupant_loc_assigned[loc_index] == 1:
                # Closest unassigned location
                unassigned_distances = [
                    (i, heuristic(firefighter_start, occupant_locations[i]))
                    for i in range(len(occupant_locations))
                    if occupant_loc_assigned[i] == 0
                ]

                if unassigned_distances:
                    loc_index = min(unassigned_distances, key=lambda x: x[1])[0]
                    print(f"  Redirected to unassigned occupant location {occupant_locations[loc_index]}")
                else:
                    print("  All locations assigned, using original location")

            # Extend the path to the selected location or use an existing path
            if occupant_loc_assigned[loc_index] == 0:
                path_extension = astar(firefighter_positions[firefighter-1],
                                    occupant_locations[loc_index],
                                    obstacles,
                                    fire_locations)
                firefighter_paths[firefighter].extend(path_extension)
                firefighter_positions[firefighter] = occupant_locations[loc_index]
                occupant_loc_assigned[loc_index] = 1
            else:
                # Use the path of the firefighter assigned to this location
                assigned_firefighter = next(
                    (i for i, pos in enumerate(firefighter_positions) if pos == occupant_locations[loc_index]),
                    None
                )
                if assigned_firefighter is not None:
                    firefighter_paths[firefighter] = firefighter_paths[assigned_firefighter]

    # Handle unassigned occupant locations
    for loc_index, assigned in enumerate(occupant_loc_assigned):
        if assigned == 0:
            # The closest firefighter to this occupant location
            distances = [(i, heuristic(firefighter_positions[i], occupant_locations[loc_index]))
                        for i in range(num_firefighters)]
            nearest_firefighter, _ = min(distances, key=lambda x: x[1])

            print(f"Occupant location {occupant_locations[loc_index]} is unassigned. Assigning to firefighter {nearest_firefighter + 1}.")

            path_extension = astar(firefighter_positions[nearest_firefighter],
                                    occupant_locations[loc_index],
                                    obstacles,
                                    fire_locations)

            firefighter_paths[nearest_firefighter].extend(path_extension)
            firefighter_positions[nearest_firefighter] = occupant_locations[loc_index]
            occupant_loc_assigned[loc_index] = 1

            print(f"Firefighter {nearest_firefighter + 1} extended path: {path_extension}")

    # Print paths
    for i, path in enumerate(firefighter_paths):
        print(f"Firefighter {i + 1} path: {path}")

    # Plot paths
    plot_firefighter_paths(binary_grid, firefighter_paths, fire_locations, occupant_locations,child_loc, elder_loc, adult_loc)