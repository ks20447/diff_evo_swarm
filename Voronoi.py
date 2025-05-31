import numpy as np
import matplotlib.pyplot as plt
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


def plot_results(map_polygon, clipped_cells, points):

    _, ax = plt.subplots(figsize=(10, 10))

    map_x, map_y = map_polygon.exterior.xy
    ax.plot(map_x, map_y, color="black", linewidth=2, label="Map Boundary")
    ax.fill(map_x, map_y, alpha=0.1, color="grey")

    for i, cell in enumerate(clipped_cells):
        if cell.geom_type == "Polygon":
            cell_x, cell_y = cell.exterior.xy
            color = plt.cm.get_cmap("viridis", len(clipped_cells))(i)
            ax.plot(cell_x, cell_y, color=color, linewidth=1, alpha=0.7)
            ax.fill(cell_x, cell_y, alpha=0.5, color=color)
        elif cell.geom_type == "MultiPolygon":
            for sub_poly in cell.geoms:
                cell_x, cell_y = sub_poly.exterior.xy
                color = plt.cm.get_cmap("viridis", len(clipped_cells))(i)
                ax.plot(cell_x, cell_y, color=color, linewidth=1, alpha=0.7)
                ax.fill(cell_x, cell_y, alpha=0.5, color=color)

    if points:
        site_coords = np.array([p.coords[0] for p in points])
        ax.plot(
            site_coords[:, 0],
            site_coords[:, 1],
            "o",
            color="red",
            markersize=5,
            label="Grid Points (Sites)",
        )

    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_title("Voronoi Partitions on Polygonal Map")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1))
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()

