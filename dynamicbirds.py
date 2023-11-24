import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np


def update_positions(G, dt=1):
    for node in G.nodes:
        pos = np.array(G.nodes[node]['pos'])
        velocity = np.array(G.nodes[node]['velocity'])
        new_pos = pos + velocity * dt
        G.nodes[node]['pos'] = tuple(new_pos)


def update_velocities(G, separation_distance=10, alignment_distance=20, cohesion_distance=30):
    for node in G.nodes:
        pos = np.array(G.nodes[node]['pos'])
        velocity = np.array(G.nodes[node]['velocity'])

        # Initialize forces
        separation_force = np.array([0.0, 0.0])
        alignment_force = np.array([0.0, 0.0])
        cohesion_force = np.array([0.0, 0.0])

        for neighbor in G.neighbors(node):
            neighbor_pos = np.array(G.nodes[neighbor]['pos'])
            neighbor_vel = np.array(G.nodes[neighbor]['velocity'])
            distance = np.linalg.norm(pos - neighbor_pos)

            if distance < separation_distance:
                separation_force += (pos - neighbor_pos)
            elif distance < alignment_distance:
                alignment_force += neighbor_vel
            elif distance < cohesion_distance:
                cohesion_force += (neighbor_pos - pos)

        # Update velocity
        new_velocity = velocity + 0.01 * separation_force + 0.01 * alignment_force + 0.01 * cohesion_force
        G.nodes[node]['velocity'] = tuple(new_velocity)


# Create graph and initialize
G = nx.Graph()
n_birds = 10

for i in range(n_birds):
    G.add_node(i, pos=(random.uniform(0, 100), random.uniform(0, 100)),
               velocity=(random.uniform(-1, 1), random.uniform(-1, 1)), status='active')

# Run simulation
for t in range(100):
    update_positions(G)
    update_velocities(G)
    pos = {node: G.nodes[node]['pos'] for node in G.nodes}

    plt.clf()
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='skyblue', font_size=18, node_size=700)
    plt.pause(0.1)

plt.show()
