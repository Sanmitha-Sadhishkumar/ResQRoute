#safest path
import numpy as np
import matplotlib.pyplot as plt
import heapq

# Load the binary grid
binary_grid = np.load('../static/show/binary_grid.npy')
grid_height, grid_width = binary_grid.shape

# Manhattan distance heuristic
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Extract obstacle positions from the binary grid
def extract_obstacles(binary_grid):
    obstacles = np.argwhere(binary_grid == 0)
    return set((int(pos[1]), int(pos[0])) for pos in obstacles)

# Check if a location is dangerous
def is_dangerous(x, y, dangerous_locations, danger_radius=30):
    for dx, dy in dangerous_locations:
        if abs(x - dx) <= danger_radius and abs(y - dy) <= danger_radius:
            return True
    return False

# A* Pathfinding Algorithm
def astar(start, exit_goal, obstacles, dangerous_locations):
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))
    came_from = {}
    g_score = {tuple(start): 0}
    f_score = {tuple(start): heuristic(start, exit_goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == exit_goal:
            path = []
            while tuple(current) in came_from:
                path.append(current)
                current = came_from[tuple(current)]
            return path[::-1]  # Return the path

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),  # Up, Down, Left, Right
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]  # Diagonals

        for dx, dy in directions:
            neighbor = (current[0] + dx, current[1] + dy)
            neighbor_tuple = tuple(neighbor)

            if (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height and neighbor_tuple not in obstacles):
                penalty = 0
                if is_dangerous(neighbor[0], neighbor[1], dangerous_locations):
                    penalty = 50  # Higher penalty for dangerous zones
                tentative_g_score = g_score[tuple(current)] + 1 + penalty

                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score[neighbor_tuple] = tentative_g_score + heuristic(neighbor, exit_goal)
                    heapq.heappush(open_set, (f_score[neighbor_tuple], neighbor_tuple))

    return []  # Return an empty list if no path is found

# Plotting all paths
def plot_path(binary_grid, occupants, dangerous_locations, exit1, exit2):
    fig, ax = plt.subplots()
    ax.imshow(binary_grid, cmap='gray', origin='lower')  # Grid
    occupant_color = 'brown'

    for o_type, occupant_data in occupants.items():
        position = occupant_data['start']
        ax.plot(position[0], position[1], 'o', color=occupant_color)  # Start position

        path_exit1 = occupant_data['path_exit1']
        if path_exit1:
            path_points = np.array(path_exit1)
            ax.plot(path_points[:, 0], path_points[:, 1], 'r--', alpha=0.5)

        path_exit2 = occupant_data['path_exit2']
        if path_exit2:
            path_points = np.array(path_exit2)
            ax.plot(path_points[:, 0], path_points[:, 1], 'b--', alpha=0.5)

    for (x, y) in dangerous_locations:
        ax.plot(x, y, 'rx', markersize=6)

    ax.plot(exit1[0], exit1[1], 'g*', markersize=10)
    ax.plot(exit2[0], exit2[1], 'm*', markersize=10)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=occupant_color, markersize=10, label='Occupant'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Exit 1'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', markersize=10, label='Exit 2'),
        plt.Line2D([0], [0], marker='x', color='r', markersize=6, label='Dangerous Locations', ls='')
    ]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Shortest safest Paths")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.savefig('../static/show/safe_path.png',bbox_inches='tight')

# Plot only the best routes
def plot_best_routes(binary_grid, occupants, dangerous_locations, exit1, exit2):
    fig, ax = plt.subplots()
    ax.imshow(binary_grid, cmap='gray', origin='lower')  # Grid
    occupant_color= 'brown'

    for o_type, occupant_data in occupants.items():
        best_path = occupant_data['path_exit1'] if occupant_data['distance_exit1'] < occupant_data['distance_exit2'] else occupant_data['path_exit2']
        position = occupant_data['start']
        ax.plot(position[0], position[1], 'o', color=occupant_color)  # Start position

        if best_path:  # Plot the best path
            path_points = np.array(best_path)
            ax.plot(path_points[:, 0], path_points[:, 1], 'b-', alpha=0.7, linewidth=2)

    for (x, y) in dangerous_locations:
        ax.plot(x, y, 'rx', markersize=6)

    ax.plot(exit1[0], exit1[1], 'g*', markersize=10)
    ax.plot(exit2[0], exit2[1], 'm*', markersize=10)

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=occupant_color, markersize=10, label='Occupant'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Exit 1'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', markersize=10, label='Exit 2'),
        plt.Line2D([0], [0], marker='x', color='r', markersize=6, label='Dangerous Locations', ls='')
    ]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Best Routes")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.savefig('../static/show/safe_best.png', bbox_inches='tight')

def safe_path_res(adult, child, elderly):
# Exit points
    exit1 = (0, 350)
    exit2 = (700, 600)

    # Extract obstacles
    obstacles = extract_obstacles(binary_grid)
    starting_positions={}
    for i,j in enumerate(adult):
        starting_positions[f'adult{i+1}'] = j
    for i,j in enumerate(child):
        starting_positions[f'child{i+1}'] = j
    for i,j in enumerate(elderly):
        starting_positions[f'elderly{i+1}'] = j
    # starting_positions = {
    #     'p1': [500, 600],
    #     'p2': [920, 220],
    #     'p3': [1000, 420],
    #     'p4': [590,200]
    # }

    dangerous_locations = set([ (400, 500), (900, 470)])

    occupants = {}

    # Calculate paths for each occupant to both exit points
    for o_type, start in starting_positions.items():
        path_to_exit1 = astar(start, exit1, obstacles, dangerous_locations)
        distance_exit1 = len(path_to_exit1)

        path_to_exit2 = astar(start, exit2, obstacles, dangerous_locations)
        distance_exit2 = len(path_to_exit2)

        occupants[o_type] = {
            'start': start,
            'path_exit1': path_to_exit1,
            'path_exit2': path_to_exit2,
            'distance_exit1': distance_exit1,
            'distance_exit2': distance_exit2
        }

    # Display paths for each occupant
    for o_type, occupant_data in occupants.items():
        print(f"{o_type} Path to Exit 1: {occupant_data['path_exit1']}")
        print(f"{o_type} Path to Exit 2: {occupant_data['path_exit2']}\n")

    # Plot the paths
    plot_path(binary_grid, occupants, dangerous_locations, exit1, exit2)
    plot_best_routes(binary_grid, occupants, dangerous_locations, exit1, exit2)