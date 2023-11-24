import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import numpy as np

# Global variable to track the number of informed agents
informed_counts = []
# behavior weights
weights = {'separation': 2.0, 'alignment': 1.0, 'cohesion': 1.5}


class Agent:
    def __init__(self, position, velocity):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.max_speed = 1.5
        self.max_force = 0.03
        self.perception = 30  # Radius of perception for neighboring agents
        self.is_informed = False  # attribute to track information status

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
        sep = self.separation(agents) * weights['separation']
        ali = self.alignment(agents) * weights['alignment']
        coh = self.cohesion(agents) * weights['cohesion']

        self.velocity += sep + ali + coh

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

    def inform(self, agents, inform_distance):
        if self.is_informed:
            for agent in agents:
                if np.linalg.norm(self.position - agent.position) < inform_distance:
                    agent.is_informed = True


def draw_agent(agent):
    color = 'red' if agent.is_informed else 'blue'
    angle = np.arctan2(agent.velocity[1], agent.velocity[0])
    return patches.RegularPolygon(agent.position, numVertices=3, radius=2, orientation=angle, color=color)


def update(frame_num, agents, ax, x_max, y_max, inform_distance):
    # Remove existing triangles
    for patch in list(ax.patches):
        patch.remove()

    # Apply behavior and update positions
    for agent in agents:
        agent.apply_behavior(agents)
        agent.update_position(x_max, y_max)
        agent.inform(agents, inform_distance)  # Spread information
        triangle = draw_agent(agent)
        ax.add_patch(triangle)

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)

    # Count and store the number of informed agents
    num_informed = sum(agent.is_informed for agent in agents)
    informed_counts.append(num_informed)


# Define plot limits
x_max, y_max = 300, 300

# Global agent variables
num_agents = 25
num_informed_initially = 5

# Initialize informed agents
agents = [Agent(np.random.rand(2) * x_max, np.random.rand(2) - 0.5) for _ in range(num_agents)]
for _ in range(num_informed_initially):
    agents[np.random.randint(len(agents))].is_informed = True


# Set up the plot
fig, ax = plt.subplots()
for agent in agents:
    triangle = draw_agent(agent)
    ax.add_patch(triangle)

# Create and start the animation
inform_distance = 10  # Distance within which agents inform each other
# Number of frames for the simulation (controls duration)
total_frames = 1000
ani = animation.FuncAnimation(fig, update, fargs=(agents, ax, x_max, y_max, inform_distance), interval=50, frames=total_frames, repeat=False)
plt.show()

# Plotting after the simulation
plt.figure()
plt.plot(informed_counts, label='Informed Agents')
plt.title('Information Spread Over Time')
plt.xlabel('Time Step')
plt.ylabel('Number of Informed Agents')
plt.axhline(y=num_agents, color='r', linestyle='-', label='Total Agents: ' + str(num_agents))
plt.axhline(y=num_informed_initially, color='g', linestyle='-', label='Initially Informed: ' + str(num_informed_initially))
plt.legend(title=f"Weights\nSeparation: {weights['separation']}\nAlignment: {weights['alignment']}\nCohesion: {weights['cohesion']}")
plt.xlim(0, total_frames)  # Set x-axis limit to match simulation duration
plt.show()


