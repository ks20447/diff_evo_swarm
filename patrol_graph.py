import numpy as np
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
import networkx as nx
from scipy.spatial import KDTree
from shapely.ops import voronoi_diagram
from shapely.geometry import Polygon, MultiPoint, Point


def create_voronoi_partitions_on_map(map_polygon, grid_points_input):

    if not isinstance(map_polygon, Polygon):
        raise TypeError("map_polygon must be a shapely.geometry.Polygon")

    sites = [Point(p) for p in grid_points_input if map_polygon.contains(Point(p))]

    if not sites:
        print("No grid points fall within the provided map polygon.")
        return [], []

    multipoint_sites = MultiPoint(sites)

    try:
        voronoi_cells_on_envelope = voronoi_diagram(
            multipoint_sites, envelope=map_polygon.envelope
        )
    except Exception as e:
        print(f"Error during Voronoi diagram generation: {e}")
        print(
            "This might happen if there are too few points (e.g., < 2) or they are collinear."
        )
        return [], sites

    clipped_voronoi_cells = []
    for cell in voronoi_cells_on_envelope.geoms:
        clipped_cell = map_polygon.intersection(cell)

        if not clipped_cell.is_empty:
            if clipped_cell.geom_type == "Polygon":
                clipped_voronoi_cells.append(clipped_cell)
            elif clipped_cell.geom_type == "MultiPolygon":
                for poly in clipped_cell.geoms:
                    clipped_voronoi_cells.append(poly)

    return clipped_voronoi_cells, sites


def voronoi_cells(map_polygon):

    minx, miny, maxx, maxy = map_polygon.bounds
    x_coords = np.linspace(minx, maxx, 6)
    y_coords = np.linspace(miny, maxy, 5)

    xbound = (minx, maxx)
    ybound = (miny, maxy)

    grid_points_input = []
    for x in x_coords:
        for y in y_coords:
            grid_points_input.append((x, y))

    clipped_cells, used_sites = create_voronoi_partitions_on_map(
        map_polygon, grid_points_input
    )

    return clipped_cells, used_sites



def generate_patrol_graph(polygon_vertices, observation_radius, grid_resolution=0.5):
    polygon = Polygon(polygon_vertices)
    minx, miny, maxx, maxy = polygon.bounds
    
    # _, patrol_points = voronoi_cells(polygon)

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


if __name__ == "__main__":
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

