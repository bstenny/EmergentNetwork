import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import time

def draw_network(G, colored_nodes):
    """ Draw the network with colored nodes. """
    plt.clf()  # Clear the current figure
    color_map = ['green' if node in colored_nodes else 'red' for node in G]
    nx.draw(G, with_labels=True, node_color=color_map)
    plt.draw()
    plt.pause(0.5)  # Pause to allow the plot to update

def spread_information(G, start_node):
    """ Spread information from the start_node to all connected nodes in the network. """
    received_info = set()  # Set to track nodes that have received information
    queue = deque([start_node])  # Queue for BFS

    while queue:
        node = queue.popleft()
        if node not in received_info:
            received_info.add(node)
            print(f"Node {node} received the information")

            # Draw the network at each step
            draw_network(G, received_info)

            for neighbor in G.neighbors(node):
                if neighbor not in received_info:
                    queue.append(neighbor)

# Create a graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 5), (4, 5)])  # Example edges

# Initialize interactive mode and initial plot
plt.ion()
draw_network(G, colored_nodes=set())

# Spread information starting from node 1
spread_information(G, start_node=1)

# Keep the plot open
plt.ioff()
plt.show()
