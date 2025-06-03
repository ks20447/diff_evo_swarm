import numpy as np
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
import networkx as nx
from scipy.spatial import KDTree

def generate_patrol_graph(polygon_vertices, observation_radius, grid_resolution=0.5):
    polygon = Polygon(polygon_vertices)
    minx, miny, maxx, maxy = polygon.bounds

    # Step 1: Grid sampling
    x = np.arange(minx, maxx, grid_resolution)
    y = np.arange(miny, maxy, grid_resolution)
    grid_points = [Point(px, py) for px in x for py in y if polygon.contains(Point(px, py))]

    uncovered = set(grid_points)
    patrol_points = []

    # Step 2: Greedy coverage with disks
    while uncovered:
        best_pt = max(uncovered, key=lambda p: sum(p.distance(o) <= observation_radius for o in uncovered))
        patrol_points.append(best_pt)
        uncovered = {p for p in uncovered if best_pt.distance(p) > observation_radius}

    coords = np.array([[p.x, p.y] for p in patrol_points])
    kdtree = KDTree(coords)
    G = nx.Graph()

    for i, pt in enumerate(patrol_points):
        G.add_node(i, pos=(pt.x, pt.y))
        indices = kdtree.query([coords[i]], k=4)[1][0][1:]  # connect to 3 nearest neighbors
        for j in indices:
            if not G.has_edge(i, j):
                dist = np.linalg.norm(coords[i] - coords[j])
                G.add_edge(i, j, weight=dist)

    # Step 4: Solve TSP for shortest patrol route
    tsp_path = nx.approximation.traveling_salesman_problem(G, weight='weight', cycle=True)
    total_length = sum(
        np.linalg.norm(coords[tsp_path[i]] - coords[tsp_path[i + 1]]) 
        for i in range(len(tsp_path) - 1)
    )

    return G, patrol_points, tsp_path, total_length


# Define polygon (square for example)
SCALE = 2
SHIFT = (-5, -4)
MAP_VERTICES = (
    np.array(
        [
            (-2, 0),
            (11, 0),
            (11, -3),
            (14, -4),
            (17, -4),
            (20, -2.5),
            (20, 8),
            (17, 8),
            (17, 13),
            (13, 13),
            (13, 9),
            (4, 9),
            (4, 6),
            (-2, 6),
            (-2, 0),
        ]
    )
    * SCALE
    + SHIFT
)
radius = 8.0

G, patrol_points, tsp_path, path_length = generate_patrol_graph(MAP_VERTICES, radius)

print(f"Total TSP patrol length: {path_length:.2f}")

# Optional plot
import matplotlib.pyplot as plt
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, node_size=50, edge_color='lightgray')
x, y = zip(*[pos[n] for n in tsp_path])
plt.plot(x, y, color='red', linewidth=2, label='TSP path')
px, py = Polygon(MAP_VERTICES).exterior.xy
plt.plot(px, py, color='black')
plt.gca().set_aspect('equal')
plt.legend()
plt.show()

