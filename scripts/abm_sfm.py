import numpy as np
import matplotlib.pyplot as plt
import heapq
from hurdle_cords import *

repulsion_strength = 2.0
repulsion_range = 1.0
time_step = 0.1

# Define speed ratios for different pedestrian types
speed_ratios = {
    'child': 0.5,
    'adult': 1.0,
    'elderly': 0.1
}

def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

def astar(start, goal, obstacles):
    open_set = []
    heapq.heappush(open_set, (0, tuple(start)))  # (f_score, position)
    came_from = {}

    g_score = {tuple(start): 0}
    f_score = {tuple(start): heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if np.array_equal(current, goal):
            path = []
            while tuple(current) in came_from:
                path.append(current)
                current = came_from[tuple(current)]
            return path[::-1]  # Return reversed path

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = np.array(current) + np.array([dx, dy])
            neighbor_tuple = tuple(neighbor)  # Convert to tuple for key usage

            # Check for out-of-bounds and obstacles
            if (0 <= neighbor[0] < 10 and 0 <= neighbor[1] < 10 and
                    list(neighbor) not in obstacles):
                tentative_g_score = g_score[tuple(current)] + 1

                if neighbor_tuple not in g_score or tentative_g_score < g_score[neighbor_tuple]:
                    came_from[neighbor_tuple] = current
                    g_score[neighbor_tuple] = tentative_g_score
                    f_score[neighbor_tuple] = tentative_g_score + heuristic(neighbor, goal)

                    # Check if the neighbor is already in open_set
                    if not any(np.array_equal(neighbor, item[1]) for item in open_set):
                        heapq.heappush(open_set, (f_score[neighbor_tuple], neighbor_tuple))

    return []

class Pedestrian:
    def __init__(self, position, destination, p_type, obstacles):
        self.position = np.array(position, dtype=float)
        self.destination = np.array(destination, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.path = astar(self.position, self.destination, obstacles)  # Get the shortest path
        self.current_target_index = 0  # Index to track the current target in the path
        self.steps_taken = 0  # Track the number of steps
        self.type = p_type
        self.speed = speed_ratios[self.type]  # Set speed based on pedestrian type
        print(f"{self.type.capitalize()} Shortest Path:", self.path)

    def update_velocity(self):
        if self.current_target_index < len(self.path):
            target = np.array(self.path[self.current_target_index])
            direction = target - self.position
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction /= norm
            self.velocity = direction * self.speed  # Adjust velocity by speed

            if norm < 0.3:  # Reached the current target, move to the next
                self.current_target_index += 1
                self.steps_taken += 1  # Increment step count
        else:
            self.velocity = np.zeros(2)

    def update_position(self):
        self.position += self.velocity * time_step

def simulate_crowd(pedestrians, obstacles):
    plt.figure(figsize=(10, 10))
    removed_pedestrians = []
    step_count = 0  # Counter to control plot updates
    
    while pedestrians:
        for p in pedestrians:
            p.update_velocity()
            p.update_position()

            # Print direction, velocity, and position of each pedestrian
            direction = (p.destination - p.position) / (np.linalg.norm(p.destination - p.position) + 1e-5)
            print(f"{p.type.capitalize()} -> Position: {p.position}, Direction: {direction}, Velocity: {p.velocity}")

            # If the pedestrian has reached the destination, remove them
            if np.linalg.norm(p.position - p.destination) < 0.3:
                print(f"{p.type.capitalize()} reached the destination in {p.steps_taken} steps.")
                removed_pedestrians.append(p)
        
        pedestrians = [p for p in pedestrians if p not in removed_pedestrians]  # Remove reached pedestrians

        plot_positions(pedestrians, obstacles)
        plt.pause(0.5)  # Increase pause to reduce load on system
        
        step_count += 1

        if not pedestrians:  # If all pedestrians have reached their destinations, stop the simulation
            break

def plot_positions(pedestrians, obstacles):
    plt.clf()

    for p in pedestrians:
        plt.plot(p.position[0], p.position[1], 'ro')
        if p.current_target_index < len(p.path):
            target = p.path[p.current_target_index]
            plt.plot(target[0], target[1], 'g*')

        # Plot the path with a dotted line
        path_points = np.array(p.path)
        plt.plot(path_points[:, 0], path_points[:, 1], 'k--', alpha=0.5)  # Dotted line for the path

    for obs in obstacles:
        plt.plot(obs[0], obs[1], 'bs', markersize=10)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title("Path Simulation using A*")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

# Example usage
pedestrians = [
    Pedestrian(position=[0, 0], destination=[8, 8], p_type='adult', obstacles=[[4, 4], [6, 6]]),
    Pedestrian(position=[0, 0], destination=[8, 8], p_type='child', obstacles=[[4, 4], [6, 6]]),
    Pedestrian(position=[0, 0], destination=[8, 8], p_type='elderly', obstacles=[[4, 4], [6, 6]])
]
result = hurd_convert('../static/show/ground floor.png')

obstacles = [
]

for i in range(len(result)):
    for j in range(len(result[i])):
        if result[i,j]==0:
            obstacles.append([i,j])

predefined_obstacles = [[4, 4], [6, 6]]
all_obstacles = obstacles + predefined_obstacles
print(all_obstacles)

simulate_crowd(pedestrians, all_obstacles)
plt.show()