"""#Best using distance
import numpy as np
import matplotlib.pyplot as plt
import heapq
import matplotlib
matplotlib.use('Agg')
from hurdle_cords import *

exit1 = (0, 350)
exit2 = (800, 700)
# Manhattan distance
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# Extract obstacle positions from the binary grid
def extract_obstacles(binary_grid):
    obstacles = np.argwhere(binary_grid == 0)
    return set((int(pos[1]), int(pos[0])) for pos in obstacles)

def astar(start, exit_goal, obstacles, grid_width, grid_height, search_radius=5):
    print('a star called')
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))
    came_from = {}
    g_score = {tuple(start): 0}  # Cost from start to the current node
    f_score = {tuple(start): heuristic(start, exit_goal)}  # Cost from start to goal

    visited = set()  # Visited nodes

    while open_set:
        current = heapq.heappop(open_set)[1]  # Get the node in open set with the lowest f_score

        if current == exit_goal:
            path = []  # Reconstruct path if goal is reached
            while tuple(current) in came_from:
                path.append(current)
                current = came_from[tuple(current)]
            return path[::-1]  # Return the path

        # Check neighboring nodes within the search radius
        for dx in range(-search_radius, search_radius + 1):
            for dy in range(-search_radius, search_radius + 1):
                if dx == 0 and dy == 0:
                    continue
                neighbor = (current[0] + dx, current[1] + dy)
                neighbor_tuple = tuple(neighbor)

                # Check if the neighbor is within bounds and not an obstacle
                if (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height and neighbor_tuple not in obstacles):
                    tentative_g_score = g_score[tuple(current)] + 1  # Calculate cost to reach neighbor

                    # Update the path if a better path is found
                    if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                        came_from[neighbor_tuple] = current
                        g_score[neighbor_tuple] = tentative_g_score
                        f_score[neighbor_tuple] = tentative_g_score + heuristic(neighbor, exit_goal)

                        if neighbor_tuple not in visited:
                            heapq.heappush(open_set, (f_score[neighbor_tuple], neighbor_tuple))
                            visited.add(neighbor_tuple)  # Mark neighbor visited
        #print(came_from)
    return []  # Return an empty list if no path is found

# Plotting
def plot_path(binary_grid, occupants):
    fig, ax = plt.subplots()
    ax.imshow(binary_grid, cmap='gray', origin='lower')  #Grid
    occupant_colors = {
        'adult': 'purple',
        'child': 'green',
        'elderly': 'brown'
    }

    # Occupant's paths
    for o_type, occupant_data in occupants.items():
        position = occupant_data['start']
        ax.plot(position[0], position[1], 'o', color=occupant_colors[o_type])  # Start position

        path_exit1 = occupant_data['path_exit1']  # Path to Exit 1
        if path_exit1:
            path_points = np.array(path_exit1)
            ax.plot(path_points[:, 0], path_points[:, 1], 'r--', alpha=0.5)

        path_exit2 = occupant_data['path_exit2']  # Path to Exit 2
        if path_exit2:
            path_points = np.array(path_exit2)
            ax.plot(path_points[:, 0], path_points[:, 1], 'b--', alpha=0.5)

    # Plot exit points
    ax.plot(exit1[0], exit1[1], 'g*', markersize=10)
    ax.plot(exit2[0], exit2[1], 'm*', markersize=10)
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=occupant_colors['adult'], markersize=10, label='Adult'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=occupant_colors['child'], markersize=10, label='Child'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=occupant_colors['elderly'], markersize=10, label='Elderly'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Exit 1'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', markersize=10, label='Exit 2')
    ]

    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_title("Shortest paths")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.savefig('../static/show/shortest_path.png')
    plt.show()

# Plot only the best routes
def plot_best_routes(binary_grid, occupants):
    fig, ax = plt.subplots()
    ax.imshow(binary_grid, cmap='gray', origin='lower')  #Grid
    occupant_colors = {
        'adult': 'purple',
        'child': 'green',
        'elderly': 'brown'
    }

    # Plot each occupant's best path
    for o_type, occupant_data in occupants.items():
        best_path = occupant_data['path_exit1'] if occupant_data['distance_exit1'] < occupant_data['distance_exit2'] else occupant_data['path_exit2']
        position = occupant_data['start']
        ax.plot(position[0], position[1], 'o', color=occupant_colors[o_type])  # Start position

        if best_path:  # Plot the best path
            path_points = np.array(best_path)
            ax.plot(path_points[:, 0], path_points[:, 1], 'b-', alpha=0.7, linewidth=2)

    # Plot exit points
    ax.plot(exit1[0], exit1[1], 'g*', markersize=10)
    ax.plot(exit2[0], exit2[1], 'm*', markersize=10)
    ax.set_title("Best Routes for Occupants")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    plt.savefig('../static/show/best_routes.png')
    plt.show()

def path_final(binary_grid):

    grid_height, grid_width = binary_grid.shape
    print(grid_height, grid_width)
# Exit points

# Extract obstacles from the binary grid
    obstacles = extract_obstacles(binary_grid)
    starting_positions = {
    'adult': [500, 600],
    'child': [920, 220],
    'elderly': [1000, 420]
    }

    occupants = {}

    result=''''''
# Calculate paths for each occupant to both exit points
    for o_type, start in starting_positions.items():
        path_to_exit1 = astar(start, exit1, obstacles, grid_height, grid_width, search_radius=5)  # Path to Exit 1
        distance_exit1 = len(path_to_exit1)  # Distance to Exit 1
        print(distance_exit1)
        path_to_exit2 = astar(start, exit2, obstacles, grid_height, grid_width, search_radius=5)  # Path to Exit 2
        distance_exit2 = len(path_to_exit2)  # Distance to Exit 2
        print(distance_exit2)
    # Store occupant data including paths and distances
        occupants[o_type] = {
        'start': start,
        'path_exit1': path_to_exit1,
        'path_exit2': path_to_exit2,
        'distance_exit1': distance_exit1,
        'distance_exit2': distance_exit2,
        'destination_exit1': path_to_exit1[-1] if path_to_exit1 else None,
        'destination_exit2': path_to_exit2[-1] if path_to_exit2 else None,
        }
        print(f"{o_type.capitalize()} Path to Exit 1: {path_to_exit1} (\nDistance: {distance_exit1}")
        result+=f"{o_type.capitalize()} Path to Exit 1: {path_to_exit2} (\nDistance: {distance_exit1}\n"
        print(f"{o_type.capitalize()} Path to Exit 2: {path_to_exit2} (\nDistance: {distance_exit2}")
        result+=f"{o_type.capitalize()} Path to Exit 2: {path_to_exit2} (\nDistance: {distance_exit2}\n"

        best_exit = 'Exit 1' if distance_exit1 < distance_exit2 else 'Exit 2'  # Determine the best exit
        print(f"Best Exit for {o_type.capitalize()}: {best_exit}\n")
        result+=f"Best Exit for {o_type.capitalize()}: {best_exit}\n"

    plot_path(binary_grid, occupants)
    plot_best_routes(binary_grid, occupants)
    return 
    
"""

#Best using distance
import numpy as np
import matplotlib.pyplot as plt
import heapq
from hurdle_cords import *

def path_final(adult, child, elderly):
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

    def astar(start, exit_goal, obstacles, search_radius=5):
        open_set = []
        heapq.heappush(open_set, (0, tuple(start)))
        came_from = {}
        g_score = {tuple(start): 0}  # Cost from start to the current node
        f_score = {tuple(start): heuristic(start, exit_goal)}  # Cost from start to goal

        visited = set()  # Visited nodes

        while open_set:
            current = heapq.heappop(open_set)[1]  # Get the node in open set with the lowest f_score

            if current == exit_goal:
                path = []  # Reconstruct path if goal is reached
                while tuple(current) in came_from:
                    path.append(current)
                    current = came_from[tuple(current)]
                return path[::-1]  # Return the path

        # Check neighboring nodes within the search radius
            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (current[0] + dx, current[1] + dy)
                    neighbor_tuple = tuple(neighbor)

                # Check if the neighbor is within bounds and not an obstacle
                    if (0 <= neighbor[0] < grid_width and 0 <= neighbor[1] < grid_height and neighbor_tuple not in obstacles):
                        tentative_g_score = g_score[tuple(current)] + 1  # Calculate cost to reach neighbor

                    # Update the path if a better path is found
                        if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                            came_from[neighbor_tuple] = current
                            g_score[neighbor_tuple] = tentative_g_score
                            f_score[neighbor_tuple] = tentative_g_score + heuristic(neighbor, exit_goal)

                            if neighbor_tuple not in visited:
                                heapq.heappush(open_set, (f_score[neighbor_tuple], neighbor_tuple))
                                visited.add(neighbor_tuple)  # Mark neighbor visited
        return []  # Return an empty list if no path is found

# Plotting
    def plot_path(binary_grid, occupants):
        fig, ax = plt.subplots()
        ax.imshow(binary_grid, cmap='gray', origin='lower')  #Grid
        occupant_colors = {
        'adult': 'purple',
        'child': 'green',
        'elderly': 'brown'
        }

    # Occupant's paths
        for o_type, occupant_data in occupants.items():
            position = occupant_data['start']
            ax.plot(position[0], position[1], 'o', color=occupant_colors[o_type[:5] if (o_type.startswith('adult') or o_type.startswith('child')) else o_type[:7]])  # Start position

            path_exit1 = occupant_data['path_exit1']  # Path to Exit 1
            if path_exit1:
                path_points = np.array(path_exit1)
                ax.plot(path_points[:, 0], path_points[:, 1], 'r--', alpha=0.5)

            path_exit2 = occupant_data['path_exit2']  # Path to Exit 2
            if path_exit2:
                path_points = np.array(path_exit2)
                ax.plot(path_points[:, 0], path_points[:, 1], 'b--', alpha=0.5)

    # Plot exit points
        ax.plot(exit1[0], exit1[1], 'g*', markersize=10)
        ax.plot(exit2[0], exit2[1], 'm*', markersize=10)
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=occupant_colors['adult'], markersize=10, label='Adult'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=occupant_colors['child'], markersize=10, label='Child'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=occupant_colors['elderly'], markersize=10, label='Elderly'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='green', markersize=10, label='Exit 1'),
            plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='magenta', markersize=10, label='Exit 2')
        ]

        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title("Shortest paths")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        plt.savefig('../static/show/shortest_path.png')

# Plot only the best routes
    def plot_best_routes(binary_grid, occupants):
        fig, ax = plt.subplots()
        ax.imshow(binary_grid, cmap='gray', origin='lower')  #Grid
        occupant_colors = {
        'adult': 'purple',
        'child': 'green',
        'elderly': 'brown'
        }

    # Plot each occupant's best path
        for o_type, occupant_data in occupants.items():
            best_path = occupant_data['path_exit1'] if occupant_data['distance_exit1'] < occupant_data['distance_exit2'] else occupant_data['path_exit2']
            position = occupant_data['start']
            ax.plot(position[0], position[1], 'o', color=occupant_colors[o_type[:5] if (o_type.startswith('adult') or o_type.startswith('child')) else o_type[:7]])  # Start position

            if best_path:  # Plot the best path
                path_points = np.array(best_path)
                ax.plot(path_points[:, 0], path_points[:, 1], 'b-', alpha=0.7, linewidth=2)

    # Plot exit points
        ax.plot(exit1[0], exit1[1], 'g*', markersize=10)
        ax.plot(exit2[0], exit2[1], 'm*', markersize=10)
        ax.set_title("Best Routes for Occupants")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        plt.savefig('../static/show/best_routes.png')

# Exit points
    exit1 = (0, 350)
    exit2 = (800, 700)

# Extract obstacles from the binary grid
    obstacles = extract_obstacles(binary_grid)
    starting_positions={}
    for i,j in enumerate(adult):
        starting_positions[f'adult{i+1}'] = j
    for i,j in enumerate(child):
        starting_positions[f'child{i+1}'] = j
    for i,j in enumerate(elderly):
        starting_positions[f'elderly{i+1}'] = j
    # starting_positions = {
    # 'adult': adult,
    # 'child': child,
    # 'elderly': elderly
    # }

    occupants = {}
    result=''''''
# Calculate paths for each occupant to both exit points
    for o_type, start in starting_positions.items():
        path_to_exit1 = astar(start, exit1, obstacles, search_radius=5)  # Path to Exit 1
        distance_exit1 = len(path_to_exit1)  # Distance to Exit 1

        path_to_exit2 = astar(start, exit2, obstacles, search_radius=5)  # Path to Exit 2
        distance_exit2 = len(path_to_exit2)  # Distance to Exit 2

    # Store occupant data including paths and distances
        occupants[o_type] = {
        'start': start,
        'path_exit1': path_to_exit1,
        'path_exit2': path_to_exit2,
        'distance_exit1': distance_exit1,
        'distance_exit2': distance_exit2,
        'destination_exit1': path_to_exit1[-1] if path_to_exit1 else None,
        'destination_exit2': path_to_exit2[-1] if path_to_exit2 else None,
        }
        path_str=''''''
        j=0
        for i in path_to_exit1:
            path_str+=str(i)
            j+=1
            if j%7==0:
                path_str+='\n'
            else:
                path_str+=','
        print(f"{o_type.capitalize()} Path to Exit 1: \n{path_str} (\nDistance: {distance_exit1}")
        result+=f"{o_type.capitalize()} Path to Exit 1: \n{path_str} (\nDistance: {distance_exit1}\n"
        path_str=''''''
        j=0
        for i in path_to_exit1:
            path_str+=str(i)
            j+=1
            if j%7==0:
                path_str+='\n'
            else:
                path_str+=','
        print(f"{o_type.capitalize()} Path to Exit 2: \n{path_str} (\nDistance: {distance_exit2}")
        result+=f"{o_type.capitalize()} Path to Exit 2: \n{path_str} (\nDistance: {distance_exit2}\n"

        best_exit = 'Exit 1' if distance_exit1 < distance_exit2 else 'Exit 2'  # Determine the best exit
        print(f"Best Exit for {o_type.capitalize()}: {best_exit}\n")
        result+=f"Best Exit for {o_type.capitalize()}: {best_exit}\n"

    plot_path(binary_grid, occupants)
    plot_best_routes(binary_grid, occupants)
    return result