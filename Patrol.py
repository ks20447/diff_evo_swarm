import math
import numpy as np
from shapely import difference
from matplotlib.path import Path
from shapely.geometry import Polygon
from matplotlib.patches import PathPatch
from shapely.affinity import rotate, translate


def rotate_point(point, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    rot_matrix = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return rot_matrix @ np.array(point)


def get_vertices_from_shape_name(shape, side, clock=True):

    match shape:
        case "Triangle":
            height = np.sqrt(3) / 2 * side
            coords = np.array(
                [
                    (0, 2 * height / 3),
                    (side / 2, -height / 3),
                    (-side / 2, -height / 3),
                ]
            )
            perimeter = 3 * side

        case "Square":
            half = side / 2
            coords = np.array(
                [
                    (-half, -half),
                    (-half, half),
                    (half, half),
                    (half, -half),
                ]
            )
            perimeter = 4 * side

        case "Hexagon":
            coords = np.array(
                [
                    (side * np.cos(-np.pi / 3 * i), side * np.sin(-np.pi / 3 * i))
                    for i in range(6)
                ]
            )
            perimeter = 6 * side

        case _:
            coords = np.array([])
            perimeter = 0

    if not clock:
        return coords[::-1], perimeter
    else:
        return coords, perimeter


class OmniPatrol:

    def __init__(self, shape, centroid, s_r, side="max", angle=0):

        self.shape = shape.capitalize()
        self.centroid = centroid
        self.s_r = s_r
        self.side_bounds = self.generate_bounds_from_shape()

        if type(side) == str:
            if side.capitalize() == "Min":
                self.side = self.side_bounds[0]
            elif side.capitalize() == "Max":
                self.side = self.side_bounds[1]
            else:
                raise ValueError("Invalid Side Value")
        else:
            self.side = side

        self.angle = angle

        self.vertices, self.perimeter = get_vertices_from_shape_name(
            self.shape, self.side
        )
        if self.vertices.size == 0:
            raise ValueError("Invalid shape type")

        self.patrol = self.create_patrol_polygon()
        self.coverage = self.create_coverage_polygon()
        self.outline = self.create_outline_polygon()

        self.observations = difference(self.outline, self.coverage)

        self.patrol = self.transform_polygon(self.patrol)
        self.coverage = self.transform_polygon(self.coverage)
        self.outline = self.transform_polygon(self.outline)

        self.observations = self.transform_polygon(self.observations)

    def create_patrol_polygon(self):

        return Polygon(self.vertices)

    def create_coverage_polygon(self):

        side = self.side

        if self.shape == "Triangle":
            extension = np.sqrt(3) / 3 * side
            offset = np.radians(30)

            base_directions = [
                np.array([np.cos(offset), np.sin(offset)]),
                np.array([np.cos(offset), -np.sin(offset)]),
                np.array([2 * np.cos(offset), -2 * np.sin(offset)]),
            ]

            coverage = []
            anchor = self.vertices[0]

            for i, vertex in enumerate(self.vertices):
                coverage.append(vertex)
                for direction in base_directions:
                    point = anchor + direction * extension
                    rotated = rotate_point(point, i * -120)
                    coverage.append(rotated)

        elif self.shape == "Square":

            coverage = self.vertices

        elif self.shape == "Hexagon":

            coverage = []

            base_offset = np.array([side / 2, side * np.sqrt(3) / 6])
            ref_point = self.vertices[0] - base_offset

            for i, vertex in enumerate(self.vertices):
                coverage.append(vertex)
                rotated = rotate_point(ref_point, i * -60)
                coverage.append(rotated)

        return Polygon(coverage)

    def create_outline_polygon(self):

        return self.patrol.buffer(self.s_r)

    def transform_polygon(self, polygon):

        polygon = rotate(polygon, self.angle, origin=(0, 0))
        polygon = translate(polygon, self.centroid[0], self.centroid[1])

        return polygon

    def generate_bounds_from_shape(self):
        lower = 1.0
        upper_bounds = {
            "Triangle": (3 * self.s_r) / (2 * np.sqrt(3)),
            "Square": 2 * self.s_r,
            "Hexagon": (2 * self.s_r) / np.sqrt(3),
        }
        return [lower, upper_bounds.get(self.shape, lower)]

    def plot(self, ax, outline=False):

        patrol_x, patrol_y = self.patrol.exterior.xy
        coverage_x, coverage_y = self.coverage.exterior.xy

        if outline:
            outline_x, outline_y = self.outline.exterior.xy
            ax.fill(outline_x, outline_y, label="Sensing Coverage", alpha=0.5)

        ax.fill(coverage_x, coverage_y, label="Triangulation Coverage")
        ax.plot(patrol_x, patrol_y, color="k", linestyle="--", label="Patrol Route")

    def plot_difference(self, ax):

        vertices = []
        codes = []

        def add_ring(ring):
            x, y = ring.xy
            verts = np.column_stack((x, y))
            verts[-1] = verts[0]  # Ensure closure
            codes_ring = (
                [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
            )
            return verts, codes_ring

        # Add exterior
        v, c = add_ring(self.observations.exterior)
        vertices.extend(v)
        codes.extend(c)

        # Add holes
        for interior in self.observations.interiors:
            v, c = add_ring(interior)
            vertices.extend(v)
            codes.extend(c)

        path = Path(vertices, codes)
        patch = PathPatch(path)
        ax.add_patch(patch)


class DirPatrol:

    def __init__(self, shape, centroid, s_r, side, angle=0, clock=True):

        self.shape = shape.capitalize()
        self.centroid = centroid
        self.s_r = s_r
        self.side = side
        self.angle = angle
        self.clock = clock

        self.vertices, self.perimeter = get_vertices_from_shape_name(
            self.shape, self.side, clock=self.clock,
        )
        if self.vertices.size == 0:
            raise ValueError("Invalid shape type")

        self.offset = self.get_offset_from_shape_name()
        self.patrol = self.create_patrol_polygon()
        self.coverages = self.create_coverage_polygons()
        self.sweeping = self.create_sweeping_polygons()

    def get_offset_from_shape_name(self):

        shape_offsets = {
            "Triangle": np.radians(30),
            "Square": np.radians(45),
            "Hexagon": np.radians(60),
        }

        return shape_offsets.get(self.shape, 0)

    def get_angle_to_vertex(self, point_start, point_end):

        x1, y1 = point_start
        x2, y2 = point_end

        return math.atan2(y2 - y1, x2 - x1)

    def create_patrol_polygon(self):

        return Polygon(self.vertices)

    def create_coverage_polygons(self):

        def coverage_box(start, end, offset):
            trajectory = self.get_angle_to_vertex(start, end) + offset
            dx = self.s_r * np.cos(trajectory)
            dy = self.s_r * np.sin(trajectory)
            return start + (dx, dy), end + (dx, dy)

        coverages = {}
        num_edges = len(self.vertices)
        vertices = self.vertices

        for edge in range(num_edges):
            start, end = vertices[edge % num_edges], vertices[(edge + 1) % num_edges]

            antenna_1 = coverage_box(start, end, self.offset)
            antenna_2 = coverage_box(start, end, -self.offset)

            coverages[f"polygon_{edge}"] = Polygon(
                np.array(
                    [
                        start,
                        antenna_1[0],
                        antenna_1[1],
                        end,
                        antenna_2[1],
                        antenna_2[0],
                    ]
                )
            )

        return coverages

    def create_sweeping_polygons(self):
        
        sweeps = {}
        num_edges = len(self.vertices)

        for ind, vertex in enumerate(self.vertices):
            start = self.vertices[(ind - 1) % num_edges]
            end = self.vertices[(ind + 1) % num_edges]

            for suffix, offset in zip(("a", "b"), (-self.offset, self.offset)):
                angle_start = (self.get_angle_to_vertex(start, vertex) + offset) % (2 * np.pi)
                angle_end = (self.get_angle_to_vertex(vertex, end) + offset) % (2 * np.pi)

                if angle_start <= angle_end:
                    angle_start += 2 * np.pi

                angles = np.linspace(angle_start, angle_end, 30)
                arc = [
                    (vertex[0] + self.s_r * np.cos(a), vertex[1] + self.s_r * np.sin(a))
                    for a in angles
                ]

                sweeps[f"sweeps_{ind}_{suffix}"] = Polygon([vertex] + arc + [vertex])

        return sweeps

