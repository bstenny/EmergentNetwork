import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

# Need to set this for reproducibility
np.random.seed(42)

# behavior weights
weights = {'separation': 2.0, 'alignment': 3.0, 'cohesion': 2.0}
max_transfer_rate = 0.1


class Agent:
    def __init__(self, position, velocity, capacity, load, demand):
        # Position and velocity attributes
        self.position = np.array(position)
        self.velocity = np.array(velocity)

        # Resource-related attributes
        self.capacity = capacity  # Maximum resource capacity
        self.load = load  # Current resource load
        self.demand = demand  # Current resource demand

        self.max_speed = 1.5
        self.max_force = 0.03
        self.perception = 30

    def calculate_demand(self):
        # Calculate met demand as min of load and demand
        met_demand = min(self.load, self.demand)

        # Unmet demand is the remaining demand
        unmet_demand = max(self.demand - self.load, 0)

        return met_demand, unmet_demand

    def update_position(self, x_max, y_max):
        # Ensure the velocity doesn't exceed the max speed
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * self.max_speed

        # Update position
        self.position += self.velocity

        # Wrap around if the agent goes out of bounds
        self.position[0] = self.position[0] % x_max
        self.position[1] = self.position[1] % y_max

    def apply_behavior(self, neighbors, agents):
        sep = self.separation(neighbors, agents)
        ali = self.alignment(neighbors, agents)
        coh = self.cohesion(neighbors, agents)
        self.velocity += sep + ali + coh

    def update_resource_state(self):
        # Generate resources
        resource_generation = min(self.capacity - self.load, self.demand)
        self.load += resource_generation

        # Consume resources
        resource_consumption = min(self.load, self.demand)
        self.load -= resource_consumption

    def interact_with_neighbors(self, neighbors, agents):
        for neighbor_idx in neighbors:
            neighbor = agents[neighbor_idx]
            # Existing logic for resource transfer
            if self.load < self.demand and neighbor.load > neighbor.demand:
                resource_needed = self.demand - self.load
                resource_available = neighbor.load - neighbor.demand
                resource_transfer = min(resource_needed, resource_available, max_transfer_rate)
                self.load += resource_transfer
                neighbor.load -= resource_transfer

    def separation(self, neighbors, agents):
        steering = np.zeros(2)
        total = 0
        for neighbor_idx in neighbors:
            neighbor = agents[neighbor_idx]
            distance = np.linalg.norm(self.position - neighbor.position)
            if distance < self.perception: # may be redundant due to network being updated based on proximity every time step
                diff = self.position - neighbor.position
                diff /= distance  # Weight by distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            steering = steering / np.linalg.norm(steering) * self.max_force * weights['separation']
        return steering

    def alignment(self, neighbors, agents):
        steering = np.zeros(2)
        total = 0
        for neighbor_idx in neighbors:
            neighbor = agents[neighbor_idx]
            steering += neighbor.velocity
            total += 1
        if total > 0:
            steering /= total
            steering = steering / np.linalg.norm(steering) * self.max_force * weights['alignment']
        return steering

    def cohesion(self, neighbors, agents):
        steering = np.zeros(2)
        total = 0
        for neighbor_idx in neighbors:
            neighbor = agents[neighbor_idx]
            if self.demand != neighbor.demand:
                steering += neighbor.position
                total += 1
        if total > 0:
            steering /= total
            steering -= self.position
            steering = steering / np.linalg.norm(steering) * self.max_force * weights['cohesion']
        return steering


class Simulation:
    def __init__(self, num_agents, x_max, y_max, lattice_size, max_capacity, max_initial_load, max_demand_ratio):
        self.agents = []
        for _ in range(num_agents):
            # Randomly determine capacity
            capacity = np.random.uniform(0.5 * max_capacity, max_capacity)

            # Set demand to not exceed a certain ratio of capacity
            max_demand = capacity * max_demand_ratio
            demand = np.random.uniform(0.1 * max_demand, max_demand)

            # Initialize the agent
            agent = Agent(
                position=np.random.rand(2) * x_max,
                velocity=np.random.rand(2) - 0.5,
                capacity=capacity,
                load=np.random.uniform(0, capacity),  # Initial load up to capacity
                demand=demand
            )
            self.agents.append(agent)

        self.network = create_lattice_network(num_agents, lattice_size)
        self.x_max = x_max
        self.y_max = y_max
        self.total_demand = 0
        self.total_met_demand = 0
        # Lists to store data for plotting
        self.met_demand_over_time = []
        self.unmet_demand_over_time = []
        self.variance_over_time = []

    def update_network_structure(self):
        proximity_threshold = 10  # Define a threshold for proximity
        for i, agent in enumerate(self.agents):
            # Recompute neighbors for each agent based on current positions
            self.network[i] = [
                j for j, other_agent in enumerate(self.agents)
                if i != j and np.linalg.norm(agent.position - other_agent.position) < proximity_threshold
            ]

    def run_simulation(self, total_frames):
        fig, ax = plt.subplots(figsize=(14, 9))
        plot_elements = [ax.add_patch(patches.RegularPolygon(agent.position, numVertices=3, radius=2, orientation=0, color='blue')) for agent in self.agents]

        def update(frame):
            ax.clear()
            ax.set_xlim(0, self.x_max)
            ax.set_ylim(0, self.y_max)
            new_plot_elements = []
            self.update_network_structure()

            # Calculate efficiency metrics for this frame
            frame_met_demand, frame_unmet_demand, frame_variance = calculate_frame_efficiency(self.agents)

            self.met_demand_over_time.append(frame_met_demand)
            self.unmet_demand_over_time.append(frame_unmet_demand)
            self.variance_over_time.append(frame_variance)

            for i, agent in enumerate(self.agents):
                neighbors = self.network[i]
                agent.apply_behavior(neighbors, self.agents)
                agent.update_position(self.x_max, self.y_max)
                agent.update_resource_state()

                # Interact with neighbors
                agent.interact_with_neighbors(neighbors, self.agents)

                # Update plot elements - triangles for each agent
                angle = np.arctan2(agent.velocity[1], agent.velocity[0])
                triangle = patches.RegularPolygon(agent.position, numVertices=3, radius=2, orientation=angle,
                                                  color='blue')
                ax.add_patch(triangle)
                new_plot_elements.append(triangle)

            # Reset demands for this frame
            frame_total_demand = 0
            frame_met_demand = 0

            # Update demand metrics
            for agent in self.agents:
                met_demand, unmet_demand = agent.calculate_demand()
                frame_total_demand += agent.demand
                frame_met_demand += met_demand

            # Accumulate to total metrics
            self.total_demand += frame_total_demand
            self.total_met_demand += frame_met_demand

            return new_plot_elements

        ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=50, repeat=False)
        plt.show()

# just creates an initial graph structure that is then reconfigured every frame
def create_lattice_network(num_agents, lattice_size):
    network = {}
    for i in range(num_agents):
        x, y = i % lattice_size, i // lattice_size
        neighbors = []

        # Left neighbor
        if x > 0: neighbors.append(i - 1)
        # Right neighbor
        if x < lattice_size - 1 and i + 1 < num_agents: neighbors.append(i + 1)
        # Upper neighbor
        if y > 0: neighbors.append(i - lattice_size)
        # Lower neighbor
        if y < lattice_size - 1 and i + lattice_size < num_agents: neighbors.append(i + lattice_size)

        network[i] = neighbors
    return network


def calculate_efficiency(agents):
    total_unmet_demand = 0
    total_met_demand = 0
    total_demand = 0
    resource_variances = []

    for agent in agents:
        met_demand, unmet_demand = agent.calculate_demand()
        total_demand += agent.demand
        total_met_demand += met_demand
        total_unmet_demand += unmet_demand
        resource_variances.append(agent.load)

    average_load = sum(resource_variances) / len(resource_variances)
    variance = sum((x - average_load) ** 2 for x in resource_variances) / len(resource_variances)

    return total_met_demand, total_unmet_demand, variance


def calculate_frame_efficiency(agents):
    frame_met_demand = 0
    frame_unmet_demand = 0
    frame_resource_loads = []

    for agent in agents:
        met_demand, unmet_demand = agent.calculate_demand()
        frame_met_demand += met_demand
        frame_unmet_demand += unmet_demand
        frame_resource_loads.append(agent.load)

    # Calculate the variance of resource loads for this frame
    average_load = sum(frame_resource_loads) / len(frame_resource_loads)
    frame_variance = sum((x - average_load) ** 2 for x in frame_resource_loads) / len(frame_resource_loads)

    return frame_met_demand, frame_unmet_demand, frame_variance


# testing helper function to list neighbors of a given agent
def list_neighbors_of_agent(simulation, agent_index):
    if agent_index < len(simulation.agents):
        neighbors = simulation.network[agent_index]
        print(f"Length of neighbors: {len(neighbors)}")
        print(f"Neighbors of Agent {agent_index}: {neighbors}")
        return neighbors
    else:
        print("Invalid agent index.")
        return []


simulation = Simulation(
    num_agents=50,
    x_max=250,
    y_max=250,
    lattice_size=10,
    max_capacity=100,
    max_initial_load=50,
    max_demand_ratio=0.75  # This should be a value between 0 and 1
)
simulation.run_simulation(total_frames=1000)

total_met_demand, total_unmet_demand, resource_variance = calculate_efficiency(simulation.agents)
print(f"Total Demand Met: {total_met_demand}")
print(f"Total Unmet Demand: {total_unmet_demand}")
print(f"Resource Variance: {resource_variance}")

# neighbors = list_neighbors_of_agent(simulation, 33) # testing helper function

# Plotting
plt.figure(figsize=(12, 6))

# Met Demand Plot
plt.subplot(1, 3, 1)
plt.plot(simulation.met_demand_over_time, label='Met Demand')
plt.title('Met Demand Over Time')
plt.xlabel('Time Step')
plt.ylabel('Met Demand')

# Unmet Demand Plot
plt.subplot(1, 3, 2)
plt.plot(simulation.unmet_demand_over_time, label='Unmet Demand')
plt.title('Unmet Demand Over Time')
plt.xlabel('Time Step')
plt.ylabel('Unmet Demand')

# Resource Variance Plot
plt.subplot(1, 3, 3)
plt.plot(simulation.variance_over_time, label='Resource Variance')
plt.title('Resource Variance Over Time')
plt.xlabel('Time Step')
plt.ylabel('Variance')

plt.tight_layout()
plt.show()


