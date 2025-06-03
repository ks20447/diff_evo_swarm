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
    num_vertices = []
    coverage_shapes = []
    observation_shapes = []
    perimeter_total = 0
    angle_array = []
    penalty = 0
    w1, w2, w3, w4, w5 = weights

    for shape, count in config.items():
        for _ in range(count):
            x = params[idx]
            y = params[idx + 1]
            angle = params[idx + 2]
            length = params[idx + 3]

            patrol = OmniPatrol(shape, (x, y), S_R, side=length, angle=angle)
            num_vertices.append(len(patrol.vertices))

            coverage_shapes.append(patrol.coverage)
            observation_shapes.append(patrol.observations)
            perimeter_total += patrol.perimeter
            angle_array.append(angle)

            if not patrol.patrol.buffer(1.0, join_style="mitre").within(map_boundary):
                penalty += 1e10

            idx += 4

    sorted_angles = sorted(angle_array)
    num_agents = len(sorted_angles)
    angle_distribution_penalty = 0

    if num_agents > 1:
        total_angle_range = 120.0

        # 1. On a circle, the ideal separation is the total range divided by the number of agents.
        #    e.g., for 3 agents: 120 / 3 = 40 degrees.
        ideal_separation = total_angle_range / num_agents

        # 2. Calculate the N separations (gaps) between the N agents.
        #    This includes the wraparound gap from the last to the first agent.
        separations = np.diff(sorted_angles)
        wraparound_separation = total_angle_range - sorted_angles[-1] + sorted_angles[0]
        
        all_separations = np.append(separations, wraparound_separation)

        # 3. Calculate the total deviation from the ideal separation.
        total_deviation = np.sum(np.abs(all_separations - ideal_separation))

        # 4. Normalize to a percentage. The maximum possible deviation is
        #    2 * (total_angle_range * (num_agents - 1) / num_agents).
        max_deviation = 2 * (total_angle_range * (num_agents - 1) / num_agents)

        if max_deviation > 0:
            angle_distribution_penalty = (total_deviation / max_deviation) * 100

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
        -w1 * (inside_coverage.area / map_boundary.area * 100)
        - w2 * (overlap_coverage * 100)
        + w3 * (overlap_observations * 100)
        + w4 * (overlap_penalty * 100)
        + w5 * (angle_distribution_penalty)
        + penalty
    )
