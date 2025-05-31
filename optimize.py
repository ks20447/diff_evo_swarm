import numpy as np
from Patrol import OmniPatrol
from shapely.ops import unary_union


def generate_bounds(config, Sr, x_bound, y_bound):
    bounds = []
    x_min, x_max = x_bound
    y_min, y_max = y_bound
    for shape, count in config.items():
        for _ in range(count):
            if shape == "Triangle":
                length_max = (3 * Sr) / (2 * np.sqrt(3))
            elif shape == "Square":
                length_max = 2 * Sr
            else:
                length_max = (2 * Sr) / np.sqrt(3)
            bounds.extend(
                [
                    (x_min, x_max),
                    (y_min, y_max),
                    (0, 360),
                    (1.0, length_max),
                ]
            )
    return bounds


# Generalized objective function
def generalized_objective(params, S_R, weights, config, map_boundary):

    idx = 0
    coverage_shapes = []
    observation_shapes = []
    perimeter_total = 0
    penalty = 0
    w1, w2, w3, w4 = weights

    for shape, count in config.items():
        for _ in range(count):
            x = params[idx]
            y = params[idx + 1]
            angle = params[idx + 2]
            length = params[idx + 3]

            patrol = OmniPatrol(shape, (x, y), S_R, side=length, angle=angle)

            coverage_shapes.append(patrol.coverage)
            observation_shapes.append(patrol.observations)
            perimeter_total += patrol.perimeter

            if not patrol.patrol.buffer(1.0, join_style="mitre").within(map_boundary):
                penalty += 1e10

            idx += 4

    # Compute combined coverage
    total_coverage = coverage_shapes[0]
    for shape in coverage_shapes[1:]:
        total_coverage = total_coverage.union(shape)

    inside_coverage = total_coverage.intersection(map_boundary)
    # wasted_coverage = total_coverage.difference(map_boundary)

    sum_of_individual_areas = sum(shape.area for shape in coverage_shapes)
    coverage_union = unary_union(coverage_shapes).area

    overlap_coverage = coverage_union / sum_of_individual_areas

    sum_of_observation_areas = sum(shape.area for shape in observation_shapes)
    difference_union = unary_union(observation_shapes).area

    overlap_observations = difference_union / sum_of_observation_areas

    coverage_union_geom = unary_union(coverage_shapes)
    observation_union_geom = unary_union(observation_shapes)

    overlap_area = coverage_union_geom.intersection(observation_union_geom).area
    normalization = coverage_union_geom.area + observation_union_geom.area

    overlap_penalty = overlap_area / normalization

    # Multi-objective scalarisation
    return (
        - w1 * (inside_coverage.area / map_boundary.area * 100)
        - w2 * (overlap_coverage * 100)
        + w3 * (overlap_observations * 100)
        + w4 * (overlap_penalty * 100)
        + penalty
    )