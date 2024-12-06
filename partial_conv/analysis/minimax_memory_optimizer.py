import heapq
from collections import defaultdict

import heapq
import math

def find_minimax_path(graph, start, end):
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

def all_paths_below_threshold(graph, start, end, threshold):
    """
    Finds all paths from start to end in a weighted graph where the maximum edge weight is below a specified threshold.

    :param graph: A 2D list (adjacency matrix) where graph[i][j] is the weight of the edge from i to j.
                  Use float('inf') for no direct edge between i and j.
    :param start: The starting node (index).
    :param end: The destination node (index).
    :param threshold: The maximum allowed edge weight along any path.
    :return: A list of tuples (path, max_weight), where `path` is the list of nodes and `max_weight` is the maximum edge weight of the path.
    """
    n = len(graph)  # Number of nodes
    results = []  # To store all valid paths
    pq = [(0, start, [start])]  # Priority queue: (max_edge_cost, current_node, path_to_node)

    while pq:
        current_max_cost, current_node, path = heapq.heappop(pq)

        # If we reach the end node and the max cost is below the threshold, save the path
        if current_node == end and current_max_cost < threshold:
            results.append((path, current_max_cost))
            continue

        # Explore neighbors
        for neighbor in range(n):
            if graph[current_node][neighbor] != float('inf') and neighbor not in path:  # Avoid revisiting nodes
                weight = graph[current_node][neighbor]
                new_max_cost = max(current_max_cost, weight)

                # Only proceed if the new max cost is below the threshold
                if new_max_cost < threshold:
                    heapq.heappush(pq, (new_max_cost, neighbor, path + [neighbor]))

    return results

# Example usage:
if __name__ == "__main__":
    # Define the graph as an adjacency matrix
    # Use math.inf to represent no direct edge
    graph = [
        [0, 4, 2, float('inf'), float('inf')],
        [4, 0, 5, 10, float('inf')],
        [2, 5, 0, 8, float('inf')],
        [float('inf'), 10, 8, 0, 6],
        [float('inf'), float('inf'), float('inf'), 6, 0]
    ]

    start = 0  # Node 'A' (index 0)
    end = 4    # Node 'E' (index 4)
    minimax_cost, minimax_path = minimax_path_matrix(graph, start, end)
    print(f"The minimax path cost from {start} to {end} is: {minimax_cost}")
    print(f"The minimax path is: {minimax_path}")

    threshold = math.inf
    paths = all_paths_below_threshold(graph, start, end, threshold)

    print(f"All paths from {start} to {end} with max weight below {threshold}:")
    for path, max_weight in paths:
        print(f"Path: {path}, Max Weight: {max_weight}")
