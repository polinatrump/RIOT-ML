import heapq
from collections import defaultdict

def minimax_path(graph, start, end):
    """
    Finds the minimax path and its cost from start to end in a weighted graph using Dijkstra's algorithm.

    :param graph: A dictionary where keys are nodes and values are lists of tuples (neighbor, weight).
    :param start: The starting node.
    :param end: The destination node.
    :return: A tuple (minimax_cost, path), where minimax_cost is the minimum of the maximum edge weights,
             and path is the list of nodes in the minimax path.
    """
    # Priority queue to store (max_edge_cost, current_node, path_to_node)
    pq = [(0, start, [start])]
    # Dictionary to track the minimum max_edge_cost to each node
    min_max_cost = {start: 0}

    while pq:
        current_max_cost, current_node, path = heapq.heappop(pq)

        # If we reach the end node, return the max_edge_cost and the path
        if current_node == end:
            return current_max_cost, path

        # Explore neighbors
        for neighbor, weight in graph[current_node]:
            # Calculate the maximum edge cost for this path
            new_max_cost = max(current_max_cost, weight)

            # If the new max cost is better than the previously recorded one, update it
            if neighbor not in min_max_cost or new_max_cost < min_max_cost[neighbor]:
                min_max_cost[neighbor] = new_max_cost
                heapq.heappush(pq, (new_max_cost, neighbor, path + [neighbor]))

    # If the end node is not reachable
    return float('inf'), []

# Example usage:
if __name__ == "__main__":
    # Define the graph as an adjacency list
    # Each edge is represented as (neighbor, weight)
    graph = {
        'A': [('B', 4), ('C', 2)],
        'B': [('A', 4), ('C', 5), ('D', 10)],
        'C': [('A', 2), ('B', 5), ('D', 8)],
        'D': [('B', 10), ('C', 8), ('E', 6)],
        'E': [('D', 6)]
    }

    start = 'A'
    end = 'E'
    minimax_cost, minimax_path = minimax_path(graph, start, end)
    print(f"The minimax path cost from {start} to {end} is: {minimax_cost}")
    print(f"The minimax path is: {minimax_path}")
