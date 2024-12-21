import random
import matplotlib.pyplot as plt
from mesa import Model
import mesa
from mesa.time import RandomActivation
from mesa.space import MultiGrid

class Person(mesa.Agent):
    """ An agent representing a person with an age group. """
    def __init__(self, unique_id, model, age_group):
        super().__init__(self, unique_id, model)  # Properly call parent constructor
        self.age_group = age_group
        self.speed = self.set_speed(age_group)

    def set_speed(self, age_group):
        """ Set speed based on age group. """
        if age_group == 'child':
            return random.uniform(1.5, 2.0)
        elif age_group == 'adult':
            return random.uniform(1.0, 1.5)
        elif age_group == 'senior':
            return random.uniform(0.5, 1.0)

    def step(self):
        """ Move the agent towards a common destination defined in the model. """
        destination = self.model.common_destination
        current_pos = self.model.grid.get_agent_position(self)  # Retrieve current position
        direction_x = destination[0] - current_pos[0]
        direction_y = destination[1] - current_pos[1]

        norm = (direction_x**2 + direction_y**2) ** 0.5
        if norm > 0:
            direction_x /= norm
            direction_y /= norm

        new_x = int(current_pos[0] + direction_x * self.speed)
        new_y = int(current_pos[1] + direction_y * self.speed)

        new_x = max(0, min(self.model.grid.width - 1, new_x))
        new_y = max(0, min(self.model.grid.height - 1, new_y))

        self.model.grid.move_agent(self, (new_x, new_y))


class CrowdModel(Model):
    """ A model with some number of agents. """

    def __init__(self, N, destination):
        super().__init__()  # Ensure proper initialization of Model
        self.num_agents = N
        self.grid = MultiGrid(10, 10, True)
        self.schedule = RandomActivation(self)
        self.common_destination = destination

        for i in range(self.num_agents):
            age_group = random.choice(['child', 'adult', 'senior'])
            agent = Person(i, self, age_group)
            self.schedule.add(agent)
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

    def step(self):
        self.schedule.step()


def visualize_model(model, steps_data, step):
    children_positions = []
    adults_positions = []
    seniors_positions = []

    for agent in model.schedule.agents:
        x, y = model.grid.get_agent_position(agent)  # Retrieve agent position
        if agent.age_group == 'child':
            children_positions.append((x, y))
        elif agent.age_group == 'adult':
            adults_positions.append((x, y))
        elif agent.age_group == 'senior':
            seniors_positions.append((x, y))

    steps_data.append((children_positions, adults_positions, seniors_positions))


def plot_steps(steps_data, common_destination, real_time=False):
    for step, (children, adults, seniors) in enumerate(steps_data):
        plt.figure(figsize=(10, 6))

        if children:
            x, y = zip(*children)
            plt.scatter(x, y, color='blue', label='Child', marker='o')
        if adults:
            x, y = zip(*adults)
            plt.scatter(x, y, color='green', label='Adult', marker='s')
        if seniors:
            x, y = zip(*seniors)
            plt.scatter(x, y, color='red', label='Senior', marker='^')

        plt.scatter(common_destination[0], common_destination[1], color='purple', s=100, label='Common Destination', edgecolor='black')

        plt.xlim(0, 9)
        plt.ylim(0, 9)
        plt.title(f'Step {step + 1}: Agent-Based Model Simulation of Age Groups')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.grid()

        if real_time:
            plt.pause(0.1)  # Real-time update
        else:
            plt.show()

    if real_time:
        plt.show()  # Show the final plot after real-time updates


def sfm_a_abm_final():
    num_agents = 100
    num_steps = 20

    common_destination = (5, 5)
    model = CrowdModel(num_agents, common_destination)

    steps_data = []

    # Choose whether to plot in real-time or after simulation
    real_time_plotting = False

    for step in range(num_steps):
        model.step()
        visualize_model(model, steps_data, step)

    # Plot results after simulation ends
    plot_steps(steps_data, common_destination, real_time=real_time_plotting)

    # Final destination count
    reached_destination_count = 0
    for agent in model.schedule.agents:
        if model.grid.get_agent_position(agent) == model.common_destination:
            reached_destination_count += 1

    print(f"Number of agents that reached the destination: {reached_destination_count} out of {num_agents}")
