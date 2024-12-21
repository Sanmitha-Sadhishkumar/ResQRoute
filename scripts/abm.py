import random
import matplotlib.pyplot as plt
from mesa import Model
import mesa
from mesa.space import MultiGrid
from mesa.time import BaseScheduler


class Person(mesa.Agent):
    """An agent representing a person with an age group."""

    def __init__(self, unique_id, model, age_group):
        self.unique_id = unique_id
        self.model = model
        self.age_group = age_group
        self.speed = self.set_speed(age_group)
        self.pos = None

    def set_speed(self, age_group):
        """Set speed based on age group."""
        if age_group == 'child':
            return random.uniform(1.5, 2.0)
        elif age_group == 'adult':
            return random.uniform(1.0, 1.5)
        elif age_group == 'senior':
            return random.uniform(0.5, 1.0)

    def step(self):
        """Move the agent towards a common destination."""
        # Check if the agent has a valid position
        if self.pos is None:
            return  # Skip if the position is not set

        destination = self.model.common_destination

        # Calculate direction towards destination
        direction_x = destination[0] - self.pos[0]
        direction_y = destination[1] - self.pos[1]

        # Normalize direction
        norm = (direction_x**2 + direction_y**2) ** 0.5
        if norm > 0:
            direction_x /= norm
            direction_y /= norm

        # Update position based on speed
        new_x = int(self.pos[0] + direction_x * self.speed)
        new_y = int(self.pos[1] + direction_y * self.speed)

        # Check boundaries
        new_x = max(0, min(self.model.grid.width - 1, new_x))
        new_y = max(0, min(self.model.grid.height - 1, new_y))

        # Move agent to the new position
        self.model.grid.move_agent(self, (new_x, new_y))


class CrowdModel(Model):
    """A model with some number of agents."""

    def __init__(self, N, destination, rng_seed=None):
        # Initialize Model with optional RNG seed
        super().__init__()
        self.random = random.Random(rng_seed)
        self.num_agents = N
        self.grid = MultiGrid(10, 10, True)
        self.schedule = BaseScheduler(self)
        self.common_destination = destination

        # Create agents
        for i in range(self.num_agents):
            age_group = self.random.choice(['child', 'adult', 'senior'])
            agent = Person(i, self, age_group)
            self.schedule.add(agent)
            # Place agent in random cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))  # Properly place the agent

    def step(self):
        """Advance the model by one step."""
        self.schedule.step()


def visualize_model(model, step):
    """Visualize the model's state at a given step."""
    plt.clf()  # Clear the current figure

    for agent in model.schedule.agents:
        if agent.pos:  # Ensure agent has a valid position
            x, y = agent.pos
            if agent.age_group == 'child':
                plt.scatter(x, y, color='blue', label='Child' if 'Child' not in plt.gca().get_legend_handles_labels()[1] else "", marker='o')
            elif agent.age_group == 'adult':
                plt.scatter(x, y, color='green', label='Adult' if 'Adult' not in plt.gca().get_legend_handles_labels()[1] else "", marker='s')
            elif agent.age_group == 'senior':
                plt.scatter(x, y, color='red', label='Senior' if 'Senior' not in plt.gca().get_legend_handles_labels()[1] else "", marker='^')

    # Plot common destination
    plt.scatter(model.common_destination[0], model.common_destination[1], color='purple', s=100, label='Common Destination', edgecolor='black')

    plt.xlim(0, model.grid.width - 1)
    plt.ylim(0, model.grid.height - 1)
    plt.title(f'Step {step + 1}: Agent-Based Model Simulation of Age Groups')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend()
    plt.grid()

    plt.pause(0.1)  # Pause for 0.1 seconds to update the plot


def abm_predict():
    # Simulation Parameters
    num_agents = 100
    num_steps = 15
    common_destination = (5, 5)
    rng_seed = 42  # Set a seed for reproducibility

    # Create and run the model
    model = CrowdModel(num_agents, common_destination, rng_seed)

    # Set up the plot for real-time visualization
    plt.figure(figsize=(10, 6))

    for step in range(num_steps):
        model.step()
        visualize_model(model, step)

    plt.show()  # Ensure the final plot remains visible

    # After the simulation loop, count agents that reached the destination
    reached_destination_count = sum(1 for agent in model.schedule.agents if agent.pos == model.common_destination)

    print(f"Number of agents that reached the destination: {reached_destination_count} out of {num_agents}")
    return f"Number of agents that reached the destination: {reached_destination_count} out of {num_agents}"