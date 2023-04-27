graph = {
    "A": ["B", "C"],
    "B": ["D", "E"],
    "C": ["F", "G"],
    "D": [],
    "E": [],
    "F": [],
    "G": []
}

start = 'A'
goal = 'E'
visited = {node: False for node in graph.keys()}

def dfs(node, visited, graph, goal):
    # Mark the current node as visited
    visited[node] = True
    print(node)

    # Check if the current node is the goal node
    if node == goal:
        print("Goal node found!")
        return True

    # Recur for all the vertices adjacent to this vertex
    for neighbor in graph[node]:
        if not visited[neighbor]:
            if dfs(neighbor, visited, graph, goal):
                return True

    return False

dfs(start, visited, graph, goal)
