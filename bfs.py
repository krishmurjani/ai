grap = {
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
open_list=[]
visited = {node: False for node in grap.keys()}

def bfs(grap, goal):
    open_list.append(start)

    visited[start] = True
    while len(open_list) > 0:
        current = open_list.pop(0)
        print(current + " ---> ", end="")
        if current == goal:
            print("found")
            break

        for neighbors in grap[current]:
            if not visited[neighbors]:
                visited[neighbors] = True
                open_list.append(neighbors)
        
bfs(grap, goal)
