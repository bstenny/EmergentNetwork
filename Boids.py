import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np


class Agent:
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.max_speed = 1.5
        self.max_force = 0.03
        self.perception = 30  # Radius of perception for neighboring agents

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
        sep = self.separation(agents)  # Separation force
        ali = self.alignment(agents)   # Alignment force
        coh = self.cohesion(agents)    # Cohesion force

        # weights
        self.velocity += sep * 1.5 + ali * 1.0 + coh * 2.0

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


def draw_agent(agent):
    """Draw an agent as a triangle oriented in the direction of its velocity."""
    angle = np.arctan2(agent.velocity[1], agent.velocity[0])
    return patches.RegularPolygon(agent.position, numVertices=3, radius=2, orientation=angle, color='blue')


def update(frame_num, agents, ax, x_max, y_max):
    # Remove existing triangles
    for patch in list(ax.patches):
        patch.remove()

    # Apply behavior and update positions
    for agent in agents:
        agent.apply_behavior(agents)
        agent.update_position(x_max, y_max)
        triangle = draw_agent(agent)
        ax.add_patch(triangle)

    # Set plot limits
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)


# Define plot limits
x_max, y_max = 300, 300

# Create agents
num_agents = 25
agents = [Agent(np.random.rand(2) * x_max, np.random.rand(2) - 0.5) for _ in range(num_agents)]

# Set up the plot
fig, ax = plt.subplots()
for agent in agents:
    triangle = draw_agent(agent)
    ax.add_patch(triangle)

# Create and start the animation
ani = animation.FuncAnimation(fig, update, fargs=(agents, ax, x_max, y_max), interval=50)
plt.show()
