import numpy as np
import matplotlib.pyplot as plt
from firebase_admin import db, credentials
from firebase_initialize import *

# Define Social Force Model parameters
repulsion_strength = 2.0
repulsion_range = 1.0
time_step = 0.1  # Time step for simulation

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

print(speed_ratios)
# Colors for different pedestrian types
pedestrian_colors = {
    'child': 'green',
    'adult': 'red',
    'elderly': 'orange'
}

class Pedestrian:
    def __init__(self, position, destination, p_type='adult'):
        # Convert position and destination to float arrays to avoid type mismatch
        self.position = np.array(position, dtype=float)
        self.destination = np.array(destination, dtype=float)
        self.velocity = np.zeros(2, dtype=float)  # Ensure velocity is float
        self.type = p_type
        self.desired_speed = speed_ratios[self.type]  # Set speed based on pedestrian type

    def update_velocity(self, pedestrians, obstacles):
        # Attractive force toward destination
        direction = self.destination - self.position
        norm = np.linalg.norm(direction)
        if norm != 0:
            direction /= norm  # Normalize direction vector
        attractive_force = self.desired_speed * direction

        # Repulsive force from obstacles
        repulsive_force = np.zeros(2)
        for obs in obstacles:
            obstacle_direction = self.position - obs
            distance = np.linalg.norm(obstacle_direction)
            if distance < repulsion_range and distance != 0:
                obstacle_direction /= distance  # Normalize
                # Repulsion strength decreases with distance
                repulsive_force += repulsion_strength * (1 / distance - 1 / repulsion_range) * obstacle_direction

        # Repulsive force from other pedestrians
        for other in pedestrians:
            if other is not self:  # Avoid self-repulsion
                pedestrian_direction = self.position - other.position
                distance = np.linalg.norm(pedestrian_direction)
                if distance < repulsion_range and distance != 0:
                    pedestrian_direction /= distance  # Normalize
                    # Repulsion strength decreases with distance
                    repulsive_force += repulsion_strength * (1 / distance - 1 / repulsion_range) * pedestrian_direction

        # Combine forces
        self.velocity = attractive_force + repulsive_force

    def update_position(self):
        self.position += self.velocity * time_step

    def get_speed(self):
        # Speed is the magnitude of the velocity vector
        return np.linalg.norm(self.velocity)

def simulate_crowd(pedestrians, obstacles, num_steps):
    plt.figure(figsize=(5, 5))
    for step in range(num_steps):
        for p in pedestrians:
            p.update_velocity(pedestrians, obstacles)  # Pass pedestrians for interaction
            p.update_position()

        plot_crowd(pedestrians, obstacles)
        plt.pause(0.1)  # Pause to create an animation effect

def plot_crowd(pedestrians, obstacles):
    plt.clf()  # Clear the current figure
    labels_plotted = {'child': False, 'adult': False, 'elderly': False}  # Track if label has been plotted

    # Plot pedestrians with different colors and labels
    for p in pedestrians:
        label = p.type.capitalize() if not labels_plotted[p.type] else ""  # Add label only once
        plt.plot(p.position[0], p.position[1], 'o', color=pedestrian_colors[p.type], label=label)
        # Annotate speed next to the pedestrian
        speed = p.get_speed()
        plt.text(p.position[0] + 0.1, p.position[1], f'Speed: {speed:.2f}', fontsize=9, color='black')
        labels_plotted[p.type] = True  # Mark this type as plotted

    # Plot obstacles
    for obs in obstacles:
        plt.plot(obs[0], obs[1], 'bs', markersize=10, label="Obstacle" if obstacles.index(obs) == 0 else "")  # Blue squares for obstacles

    # Check if the legend is needed
    if plt.gca().get_legend() is None:
        plt.legend(loc='upper left')

    plt.xlim(0, 10)
    plt.ylim(0, 10)

def social_forces_predict():
    pedestrians = [
    Pedestrian(position=[0, 0], destination=[8, 8], p_type='child'),
    Pedestrian(position=[1, 1], destination=[8, 8], p_type='adult'),
    Pedestrian(position=[2, 2], destination=[8, 8], p_type='elderly'),
    Pedestrian(position=[0.5, 0], destination=[8, 8], p_type='child'),
    Pedestrian(position=[1.5, 1.5], destination=[8, 8], p_type='adult')
    ]

    obstacles = [
    [4, 4], 
    [6, 6], 
    [5, 3], 
    ]

    simulate_crowd(pedestrians, obstacles, num_steps=100)
    plt.savefig('../static/sfm.png')