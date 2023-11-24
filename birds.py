import networkx as nx
import matplotlib.pyplot as plt
import random

# Create an undirected graph
G = nx.Graph()

# Number of birds
n_birds = 10

# Add nodes to the graph with initial random positions and velocities
for i in range(n_birds):
    G.add_node(i, pos=(random.uniform(0, 100), random.uniform(0, 100)), velocity=(random.uniform(-1, 1), random.uniform(-1, 1)), status='active')

# Add edges to represent interactions between birds within a certain distance (for demo, we'll randomly connect them)
for i in range(n_birds):
    for j in range(i + 1, n_birds):
        distance = random.uniform(10, 50)  # Replace with actual distance calculation
        if distance < 30:  # Threshold distance for adding an edge
            G.add_edge(i, j, distance=distance, interaction_level=random.uniform(0, 1))

# For visualization, use positions as layout
pos = nx.get_node_attributes(G, 'pos')

# Draw the graph
nx.draw(G, pos, with_labels=True, font_weight='bold', node_color='skyblue', font_size=18, node_size=700)

plt.show()
