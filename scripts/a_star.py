import numpy as np
import matplotlib.pyplot as plt
import heapq

repulsion_strength = 2.0
repulsion_range = 1.0
time_step = 0.1

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
    def __init__(self, position, destination, obstacles):
        self.position = np.array(position, dtype=float)
        self.destination = np.array(destination, dtype=float)
        self.velocity = np.zeros(2, dtype=float)
        self.path = astar(self.position, self.destination, obstacles)  # Get the shortest path
        self.current_target_index = 0  # Index to track the current target in the path
        print("Shortest Path:", self.path)

    def update_velocity(self):
        if self.current_target_index < len(self.path):
            target = np.array(self.path[self.current_target_index])
            direction = target - self.position
            norm = np.linalg.norm(direction)
            if norm != 0:
                direction /= norm
            self.velocity = direction

            if norm < 0.3:
                if not np.array_equal(target, self.destination):  # Check if the current target is not the destination
                    self.current_target_index += 1
                if np.array_equal(self.position, self.destination):
                    self.velocity = np.zeros(2)
        else:
            self.velocity = np.zeros(2)

    def update_position(self):
        self.position += self.velocity * time_step

def simulate_crowd(pedestrians, obstacles):
    plt.figure(figsize=(10, 10))
    while True:
        for p in pedestrians:
            p.update_velocity()
            p.update_position()

        plot_positions(pedestrians, obstacles)
        plt.pause(0.1)

        # Check if the pedestrian has reached the destination
        if np.array_equal(pedestrians[0].position, pedestrians[0].destination):
            break

def plot_positions(pedestrians, obstacles):
    plt.clf()

    for p in pedestrians:
        plt.plot(p.position[0], p.position[1], 'ro')
        if p.current_target_index < len(p.path):
            target = p.path[p.current_target_index]
            plt.plot(target[0], target[1], 'g*')

    for obs in obstacles:
        plt.plot(obs[0], obs[1], 'bs', markersize=10)

    plt.xlim(0, 10)
    plt.ylim(0, 10)
    plt.title("Path Simulation using A*")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")

def a_star_final():
# Example usage
    pedestrians = [
    Pedestrian(position=[0, 0], destination=[8, 8], obstacles=[[4, 4], [6, 6]]),
    ]

    obstacles = [
    [4, 4],
    [6, 6],
    ]

    simulate_crowd(pedestrians, obstacles)
    plt.show()
