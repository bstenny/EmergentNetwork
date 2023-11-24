import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

# Global variable to track the number of informed agents
informed_counts = []
# behavior weights
weights = {'separation': 1.5, 'alignment': 1.0, 'cohesion': 2.0}


class Agent:
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.max_speed = 1.5
        self.max_force = 0.03
        self.perception = 30  # Radius of perception for neighboring agents
        self.is_informed = False  # attribute to track information status
        self.is_informed_predator = False  # attribute for predator information

    def update_position(self, x_max, y_max):
        # Ensure the velocity doesn't exceed the max speed
        if np.linalg.norm(self.velocity) > self.max_speed:
            self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * self.max_speed

        # Update position
        self.position += self.velocity

        # Wrap around if the agent goes out of bounds
        self.position[0] = self.position[0] % x_max
        self.position[1] = self.position[1] % y_max

    def apply_behavior(self, agents):
        # Calculate the basic behavior forces
        sep = self.separation(agents) * weights['separation']
        ali = self.alignment(agents) * weights['alignment']
        coh = self.cohesion(agents) * weights['cohesion']

        # If the agent is informed about the predator, increase the speed to 90% of max speed
        if self.is_informed_predator:
            self.velocity += sep + ali + coh
        else:
            # Apply normal behavior otherwise
            self.velocity += sep + ali + coh
            # Limit the velocity to max speed
            if np.linalg.norm(self.velocity) > self.max_speed:
                self.velocity = (self.velocity / np.linalg.norm(self.velocity)) * self.max_speed

    def separation(self, agents):
        steering = np.zeros(2)
        total = 0
        for agent in agents:
            distance = np.linalg.norm(self.position - agent.position)
            if self != agent and distance < self.perception:
                diff = self.position - agent.position
                diff /= distance  # Weight by distance
                steering += diff
                total += 1
        if total > 0:
            steering /= total
            steering = steering / np.linalg.norm(steering) * self.max_force
        return steering

    def alignment(self, agents):
        steering = np.zeros(2)
        total = 0
        for agent in agents:
            if self != agent and np.linalg.norm(self.position - agent.position) < self.perception:
                steering += agent.velocity
                total += 1
        if total > 0:
            steering /= total
            steering = steering / np.linalg.norm(steering) * self.max_force
        return steering

    def cohesion(self, agents):
        steering = np.zeros(2)
        total = 0
        for agent in agents:
            if self != agent and np.linalg.norm(self.position - agent.position) < self.perception:
                steering += agent.position
                total += 1
        if total > 0:
            steering /= total
            steering -= self.position
            steering = steering / np.linalg.norm(steering) * self.max_force
        return steering

    def inform(self, agents, inform_distance, predator_distance):
        # Inform about predator first
        if self.is_informed_predator:
            for agent in agents:
                if np.linalg.norm(self.position - agent.position) < predator_distance and not agent.is_informed_predator:
                    agent.is_informed_predator = True
                    agent.is_informed = False  # No longer just regularly informed
        # Then inform about other information
        elif self.is_informed:
            for agent in agents:
                if np.linalg.norm(self.position - agent.position) < inform_distance and not agent.is_informed and not agent.is_informed_predator:  # Don't inform predator-informed agents
                    agent.is_informed = True


def draw_agent(agent):
    # Prioritize the predator-informed state
    if agent.is_informed_predator:
        color = 'orange'
    elif agent.is_informed:
        color = 'red'
    else:
        color = 'blue'
    angle = np.arctan2(agent.velocity[1], agent.velocity[0])
    return patches.RegularPolygon(agent.position, numVertices=3, radius=2, orientation=angle, color=color)


def update(frame_num, agents, ax, x_max, y_max, inform_distance, predator_distance):
    # Remove existing triangles
    for patch in list(ax.patches):
        patch.remove()

    # Apply behavior and update positions
    for agent in agents:
        agent.apply_behavior(agents)
        agent.update_position(x_max, y_max)
        agent.inform(agents, inform_distance, predator_distance)  # Use the correct method signature
        triangle = draw_agent(agent)
        ax.add_patch(triangle)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Count and store the number of informed agents
    num_informed = sum(agent.is_informed for agent in agents)
    num_informed_predator = sum(agent.is_informed_predator for agent in agents)
    informed_counts.append((num_informed, num_informed_predator))


# Define plot limits
x_max, y_max = 300, 300

# Global agent variables
num_agents = 25
num_informed_initially = 5

# Initialize agents
agents = [Agent(np.random.rand(2) * x_max, np.random.rand(2) - 0.5) for _ in range(num_agents)]

# Randomly inform a few agents
informed_indices = np.random.choice(range(num_agents), num_informed_initially, replace=False)
for idx in informed_indices:
    agents[idx].is_informed = True

# Initialize a single predator-informed agent, making sure it's not already informed
predator_informed_index = np.random.choice(list(set(range(num_agents)) - set(informed_indices)))
agents[predator_informed_index].is_informed_predator = True

# Set up the plot
fig, ax = plt.subplots()
for agent in agents:
    triangle = draw_agent(agent)
    ax.add_patch(triangle)

# Create and start the animation
inform_distance = 10  # Distance within which agents inform each other
# Number of frames for the simulation (controls duration)
total_frames = 1000
predator_distance = 10  # Distance within which agents are informed of the predator
ani = animation.FuncAnimation(fig, update, fargs=(agents, ax, x_max, y_max, inform_distance, predator_distance), interval=50, frames=total_frames, repeat=False)
plt.show()

# Plotting after the simulation
plt.figure(figsize=(10, 6))

# Adjust the count for regularly informed agents
# They should not be counted if they become predator informed
regularly_informed = [count[0] for count in informed_counts]
predator_informed = [count[1] for count in informed_counts]
# Subtract predator informed from regularly informed for each time step
adjusted_regularly_informed = [ri - pi if ri > pi else 0 for ri, pi in zip(regularly_informed, predator_informed)]

plt.plot(adjusted_regularly_informed, label='Regularly Informed Agents', color='red')
plt.plot(predator_informed, label='Predator Informed Agents', color='orange')

# Add horizontal lines to indicate the total number of agents and initially informed agents
plt.axhline(y=num_agents, color='r', linestyle='--', label='Total Agents: ' + str(num_agents))
plt.axhline(y=num_informed_initially, color='g', linestyle='--', label='Initially Regularly Informed: ' + str(num_informed_initially))

# Add title and labels
plt.title('Information Spread Over Time')
plt.xlabel('Time Step')
plt.ylabel('Number of Informed Agents')

# Set the x-axis limit to match the simulation duration
plt.xlim(0, total_frames)

# Add a legend
plt.legend(loc='upper left')

# Display the plot
plt.show()



