import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy.spatial import KDTree
import time

# Constants defining the domain
X_MAX = 500
Y_MAX = 500
X_MIN = 0
Y_MIN = 0


def update_positions_and_velocities(G, dt=1):
    for node in G.nodes:
        node_data = G.nodes[node]
        # Extract current position and velocity from the node data
        pos = np.array(node_data['pos'])
        velocity = np.array(node_data['velocity'])

        # Predicted new position
        new_pos = pos + velocity * dt

        # Check for collision with walls and reverse velocity if necessary
        if new_pos[0] >= X_MAX or new_pos[0] <= X_MIN:
            velocity[0] *= -1
            new_pos[0] = max(min(new_pos[0], X_MAX), X_MIN)  # Keep within bounds
        if new_pos[1] >= Y_MAX or new_pos[1] <= Y_MIN:
            velocity[1] *= -1
            new_pos[1] = max(min(new_pos[1], Y_MAX), Y_MIN)  # Keep within bounds

        # Update position and velocity in the graph
        node_data['pos'] = tuple(new_pos)
        node_data['velocity'] = tuple(velocity)


def update_velocities(G, separation_distance=10, alignment_distance=20, cohesion_distance=30, speed_factor=1):
    for node in G.nodes:
        pos = np.array(G.nodes[node]['pos'])
        velocity = np.array(G.nodes[node]['velocity'])

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

        total_force = 0.01 * (separation_force + alignment_force + cohesion_force)
        new_velocity = velocity + total_force

        # Normalize the velocity to maintain speed
        G.nodes[node]['velocity'] = tuple(new_velocity / np.linalg.norm(new_velocity) * speed_factor)

# def update_edges(G, distance_threshold=20):
#     positions = [G.nodes[i]['pos'] for i in G.nodes]
#     tree = KDTree(positions)
#
#     # Find pairs within the distance_threshold
#     pairs = tree.query_pairs(distance_threshold)
#
#     # Remove all existing edges
#     edges_to_remove = list(G.edges())
#     G.remove_edges_from(edges_to_remove)
#
#     # Add new edges
#     G.add_edges_from(pairs)


def update_edges(G, distance_threshold=20, tolerance=1e-9):
    # Remove all existing edges
    edges_to_remove = list(G.edges())
    G.remove_edges_from(edges_to_remove)

    # Add edges based on new positions
    for i in G.nodes():
        for j in G.nodes():
            if i >= j:
                continue
            pos_i = np.array(G.nodes[i]['pos'])
            pos_j = np.array(G.nodes[j]['pos'])
            distance = np.linalg.norm(pos_i - pos_j)

            # Adding a tolerance check
            if abs(distance - distance_threshold) <= tolerance or distance < distance_threshold:
                G.add_edge(i, j)


G = nx.Graph()
n_birds = 8

for i in range(n_birds):
    G.add_node(i, pos=(random.uniform(0, 100), random.uniform(0, 100)),
               velocity=(random.uniform(-1, 1), random.uniform(-1, 1)), status='active')

for i in range(100):
    start_time = time.time()

    # Use new function to update both positions and handle wall collisions
    update_positions_and_velocities(G)

    update_velocities(G)
    update_edges(G)
    pos = {node: G.nodes[node]['pos'] for node in G.nodes}

    plt.clf()
    nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='skyblue', font_size=18, node_size=700)
    plt.pause(0.1)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Iteration {i} took {elapsed_time} seconds")


plt.show()
