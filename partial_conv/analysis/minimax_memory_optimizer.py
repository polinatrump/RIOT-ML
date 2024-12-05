import heapq
from collections import defaultdict

import heapq
import math

def minimax_path_matrix(graph, start, end):
    """
    Finds the minimax path and its cost from start to end in a weighted graph using an adjacency matrix.

    :param graph: A 2D list (adjacency matrix) where graph[i][j] is the weight of the edge from i to j.
                  Use float('inf') for no direct edge between i and j.
    :param start: The starting node (index).
    :param end: The destination node (index).
    :return: A tuple (minimax_cost, path), where minimax_cost is the minimum of the maximum edge weights,
             and path is the list of nodes in the minimax path.
    """
    n = len(graph)  # Number of nodes
    pq = [(0, start, [start])]  # Priority queue: (max_edge_cost, current_node, path_to_node)
    min_max_cost = [math.inf] * n  # Minimum max edge cost to each node
    min_max_cost[start] = 0

    while pq:
        current_max_cost, current_node, path = heapq.heappop(pq)

        # If we reach the end node, return the max_edge_cost and the path
        if current_node == end:
            return current_max_cost, path

        # Explore neighbors
        for neighbor in range(n):
            if graph[current_node][neighbor] != math.inf:  # If there's an edge
                weight = graph[current_node][neighbor]
                new_max_cost = max(current_max_cost, weight)

                # If the new max cost is better, update it
                if new_max_cost < min_max_cost[neighbor]:
                    min_max_cost[neighbor] = new_max_cost
                    heapq.heappush(pq, (new_max_cost, neighbor, path + [neighbor]))

    # If the end node is not reachable
    return math.inf, []

# Example usage:
if __name__ == "__main__":
    # Define the graph as an adjacency matrix
    # Use math.inf to represent no direct edge
    graph = [
        [0, 12, 2, math.inf, math.inf],
        [4, 0, 5, 10, math.inf],
        [2, 5, 0, 8, math.inf],
        [math.inf, 10, 8, 0, 6],
        [math.inf, math.inf, math.inf, math.inf, math.inf]
    ]

    start = 0  # Node 'A' (index 0)
    end = 4    # Node 'E' (index 4)
    minimax_cost, minimax_path = minimax_path_matrix(graph, start, end)
    print(f"The minimax path cost from {start} to {end} is: {minimax_cost}")
    print(f"The minimax path is: {minimax_path}")
