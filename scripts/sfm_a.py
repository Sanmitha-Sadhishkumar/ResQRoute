import numpy as np
import matplotlib.pyplot as plt
import heapq
from firebase_admin import db, credentials
from firebase_initialize import *

# Social Force parameters
repulsion_strength = 2.0
repulsion_range = 1.0
time_step = 0.1
arrival_threshold = 0.1  # Distance threshold to consider arrival
result=''''''
# Speed ratios for different pedestrian types
ref = db.reference('/')

data = ref.get()
# Speed ratios for different types of pedestrians
speed = {
    'child': data.get('children'),
    'adult': data.get('adult'),
    'elderly': data.get('elderly')
}

# Calculate speed ratios relative to 'adult'
adult_speed = speed['adult']

speed_ratios = {
    'child': speed['child'] / adult_speed,
    'elderly': speed['elderly'] / adult_speed,
    'adult': 1.0
}
# Define pedestrian colors
pedestrian_colors = {
    'child': 'green',
    'adult': 'red',
    'elderly': 'orange'
}

# A* Heuristic function
def heuristic(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# A* Search algorithm
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

    return []  # Return an empty list if no path found

# Pedestrian class with Social Forces and A* Search integration
class Pedestrian:
    def __init__(self, position, destination, p_type, obstacles):
        self.position = np.array(position, dtype=float)
        self.destination = np.array(destination, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.path = astar(self.position, self.destination, obstacles)  # Get the shortest path
        self.current_target_index = 0  # Index to track the current target in the path
        self.steps_taken = 0  # Track steps taken
        self.type = p_type
        self.speed = speed_ratios[self.type]  # Set speed based on pedestrian type
        self.reached_destination = False
        print(f"{self.type.capitalize()} Shortest Path:", self.path)

    def update_velocity(self, pedestrians, obstacles):
        global result
        if self.reached_destination:
            return  # Stop updating if the pedestrian has reached the destination

        # Initialize attractive force
        attractive_force = np.zeros(2)

        # Calculate attractive force towards the next point in the path
        if self.current_target_index < len(self.path):
            target = np.array(self.path[self.current_target_index])
            direction = target - self.position
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction /= norm
            attractive_force = self.speed * direction
            if norm < 0.3:
                self.current_target_index += 1
                self.steps_taken += 1

        # Repulsive forces from obstacles
        repulsive_force = np.zeros(2)
        for obs in obstacles:
            obstacle_direction = self.position - np.array(obs)
            distance = np.linalg.norm(obstacle_direction)
            if distance < repulsion_range and distance != 0:
                obstacle_direction /= distance
                repulsive_force += repulsion_strength * (1 / distance - 1 / repulsion_range) * obstacle_direction

        # Repulsive forces from other pedestrians
        for other in pedestrians:
            if other is not self:
                pedestrian_direction = self.position - other.position
                distance = np.linalg.norm(pedestrian_direction)
                if distance < repulsion_range and distance != 0:
                    pedestrian_direction /= distance
                    repulsive_force += repulsion_strength * (1 / distance - 1 / repulsion_range) * pedestrian_direction

        # Combine forces to determine velocity
        self.velocity = attractive_force + repulsive_force

    def update_position(self):
        if not self.reached_destination:
            self.position += self.velocity * time_step

            # Check if the pedestrian has reached the destination
            if np.linalg.norm(self.destination - self.position) < arrival_threshold:
                self.reached_destination = True

# Simulation and plotting functions
def simulate_crowd(pedestrians, obstacles, num_steps=100):
    global result
    plt.figure(figsize=(10, 10))
    for step in range(num_steps):
        all_reached = True
        for p in pedestrians:
            if not p.reached_destination:
                all_reached = False
                p.update_velocity(pedestrians, obstacles)
                p.update_position()

        plot_crowd(pedestrians, obstacles)
        plt.pause(0.1)

        if all_reached:
            break

    # Print number of steps taken by each pedestrian
    for p in pedestrians:
        result+=f'{p.type.capitalize()} took {p.steps_taken} steps to reach the destination.\n'
        print(f'{p.type.capitalize()} took {p.steps_taken} steps to reach the destination.')

def plot_crowd(pedestrians, obstacles):
    plt.clf()  # Clear the current figure
    labels_plotted = {'child': False, 'adult': False, 'elderly': False}

    for p in pedestrians:
        label = p.type.capitalize() if not labels_plotted[p.type] else ""
        plt.plot(p.position[0], p.position[1], 'o', color=pedestrian_colors[p.type], label=label)
        labels_plotted[p.type] = True

        # Plot path with dotted lines
        if p.path:
            path_points = np.array(p.path)
            plt.plot(path_points[:, 0], path_points[:, 1], 'k--', alpha=0.5)

    for obs in obstacles:
        plt.plot(obs[0], obs[1], 'bs', markersize=10, label="Obstacle" if obstacles.index(obs) == 0 else "")

    if plt.gca().get_legend() is None:
        plt.legend(loc='upper left')

    plt.xlim(0, 10)
    plt.ylim(0, 10)

def sfm_a_final():
    global result
# Initialize pedestrians
    pedestrians = [
    Pedestrian(position=[0, 0], destination=[8, 8], p_type='child', obstacles=[[4, 4], [6, 6]]),
    Pedestrian(position=[0, 0], destination=[8, 8], p_type='adult', obstacles=[[4, 4], [6, 6]]),
    Pedestrian(position=[0, 0], destination=[8, 8], p_type='elderly', obstacles=[[4, 4], [6, 6]])
    ]

# Define obstacles
    obstacles = [
    [4, 4],
    [6, 6]
    ]

# Simulate the crowd
    simulate_crowd(pedestrians, obstacles, num_steps=100)
    plt.show()
    return result
