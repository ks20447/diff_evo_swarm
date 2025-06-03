import csv
import math
import time
import numpy as np
import pandas as pd
from Robot import Robot
from Patrol import OmniPatrol
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from matplotlib.colors import Normalize
from scipy.optimize import differential_evolution
from matplotlib.collections import LineCollection
from optimize import generate_bounds, generalized_objective
from Voronoi import create_voronoi_partitions_on_map, plot_results

# --- Configuration ---
SIMULATION_STEPS = 250
ORIENTATION_LINE_LENGTH = 0.3

C = 3e8
SIGNAL_SOURCE_POS = np.array([12.5, 8.1])
EMITTER_PARAMS = {"efficiency": 1.0, "pt": 30, "gain": 1.0, "wavelength": C / 2.4e9}

# Trail Color Configuration
MIN_SIGNAL_FOR_COLORMAP = -50
MAX_SIGNAL_FOR_COLORMAP = 10.0  # Adjusted based on typical values, might need tuning
MAX_PATH_HISTORY_LENGTH = 0
SELECTED_CMAP = plt.cm.viridis

SHAPE_CONFIG = {"Triangle": 3, "Square": 0, "Hexagon": 0}
S_R = 8.0
DE_WEIGHTS = [2.0, 1.0, 0.5, 1.0, 10.0]
NUM_ROBOTS = sum(SHAPE_CONFIG.values())

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

CSV_FILE = "simulation_data.csv"


def voronoi_cells(map_polygon):

    minx, miny, maxx, maxy = map_polygon.bounds
    x_coords = np.linspace(minx, maxx, 4)
    y_coords = np.linspace(miny, maxy, 4)

    xbound = (minx, maxx)
    ybound = (miny, maxy)

    grid_points_input = []
    for x in x_coords:
        for y in y_coords:
            grid_points_input.append((x, y))

    clipped_cells, used_sites = create_voronoi_partitions_on_map(
        map_polygon, grid_points_input
    )
    plot_results(map_polygon, clipped_cells, used_sites)

    return clipped_cells, xbound, ybound


def area_coverage_optimiser(cell, xbound, ybound):

    patrols = []
    bounds = generate_bounds(SHAPE_CONFIG, S_R, xbound, ybound)

    print("Optimizer Started")

    result = differential_evolution(
        generalized_objective,
        bounds,
        args=(S_R, DE_WEIGHTS, SHAPE_CONFIG, cell),
        maxiter=500,
        polish=True,
        init="halton",
        updating="deferred",
        workers=-1,
    )

    params = result.x

    print("Optimizer finished")

    idx = 0
    for shape, count in SHAPE_CONFIG.items():
        for _ in range(count):
            x = params[idx]
            y = params[idx + 1]
            angle = params[idx + 2]
            length = params[idx + 3]

            patrol = OmniPatrol(shape, (x, y), S_R, side=length, angle=angle)
            patrols.append(patrol)
            idx += 4

    return params, patrols


def plot_area_coverage(cell, patrols):

    _, ax = plt.subplots()

    x, y = cell.exterior.xy
    ax.axis("equal")
    ax.plot(x, y)

    for patrol in patrols:

        patrol.plot(ax, outline=True)

    plt.show()


def initialize_robots(robot_waypoints):
    robots = []
    source_info = {"pos": SIGNAL_SOURCE_POS, "params": EMITTER_PARAMS}
    for i in range(NUM_ROBOTS):
        if i < len(robot_waypoints):
            waypoints = robot_waypoints[i]
            # initial_pos defaults to waypoints[0] in Robot class if None
            initial_pos = waypoints[0] if len(waypoints) > 0 else None
        else:
            print(
                f"Warning: Not enough waypoint sets defined. Using default for Robot {i}"
            )
            waypoints = np.array(
                [[float(i + 1), float(i + 1)], [float(i + 2), float(i + 2)]]
            )
            initial_pos = waypoints[0]

        # Pass source_info and MAX_PATH_HISTORY_LENGTH to the Robot constructor
        robots.append(
            Robot(
                robot_id=i,
                waypoints=waypoints,
                source=source_info,
                initial_pos=initial_pos,
                max_history=MAX_PATH_HISTORY_LENGTH,
                # speed can be passed here to override ROBOT_SPEED from robot_class.py
            )
        )
    return robots


def initialize_plot(ax, cell):

    ax.plot(MAP_VERTICES[:, 0], MAP_VERTICES[:, 1], "k-", label="Map Boundary")
    ax.plot(
        SIGNAL_SOURCE_POS[0],
        SIGNAL_SOURCE_POS[1],
        "yx",
        markersize=10,
        markeredgewidth=1.5,
        label="Signal Source",
    )
    ax.plot(cell.exterior.xy[0], cell.exterior.xy[1], color="r")


def add_robots_to_plot(ax):

    robot_plots = []
    waypoint_plots_markers = []
    waypoint_plots_lines = []
    robot_labels = []
    orientation_lines_plots = []
    robot_trail_collections = []
    colors = ["r", "g", "b", "c", "m", "y", "orange"]
    norm = Normalize(vmin=MIN_SIGNAL_FOR_COLORMAP, vmax=MAX_SIGNAL_FOR_COLORMAP)

    for i in range(NUM_ROBOTS):
        robot_marker_color = colors[i % len(colors)]
        (plot,) = ax.plot(
            [],
            [],
            marker="o",
            color=robot_marker_color,
            markersize=8,
            label=f"Robot {i}",
        )
        robot_plots.append(plot)

        (orientation_line,) = ax.plot(
            [], [], linestyle="-", color=robot_marker_color, linewidth=2
        )
        orientation_lines_plots.append(orientation_line)

        trail_collection = LineCollection([], cmap=SELECTED_CMAP, norm=norm)
        trail_collection.set_linewidth(3)
        ax.add_collection(trail_collection)
        robot_trail_collections.append(trail_collection)

        if i < len(robot_waypoints) and robot_waypoints[i].size > 0:
            wps = robot_waypoints[i]
            (wp_line,) = ax.plot(
                wps[:, 0],
                wps[:, 1],
                linestyle="--",
                color=robot_marker_color,
                alpha=0.3,
            )
            waypoint_plots_lines.append(wp_line)
            (wp_marker,) = ax.plot(
                wps[:, 0],
                wps[:, 1],
                marker="x",
                color=robot_marker_color,
                markersize=7,
                alpha=0.7,
                linestyle="None",
            )
            waypoint_plots_markers.append(wp_marker)

        label = ax.text(0, 0, "", fontsize=8, color=robot_marker_color, va="bottom")
        robot_labels.append(label)

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Robot Swarm Waypoint Patrol Simulation")
    handles, legend_labels_list = ax.get_legend_handles_labels()
    unique_labels_dict = {}
    for handle, label_text in zip(handles, legend_labels_list):
        if label_text not in unique_labels_dict:
            unique_labels_dict[label_text] = handle
    ax.legend(
        unique_labels_dict.values(),
        unique_labels_dict.keys(),
        loc="upper left",
        fontsize="small",
    )

    ax.axis("equal")
    ax.grid(True)

    if NUM_ROBOTS > 0:
        sm = plt.cm.ScalarMappable(cmap=SELECTED_CMAP, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(
            sm,
            ax=ax,
            label="Signal Strength",
            orientation="vertical",
            pad=0.02,
            aspect=30,
        )

    all_points_for_limits = list(MAP_VERTICES)
    for wps_set in robot_waypoints:
        if wps_set.size > 0:
            all_points_for_limits.extend(wps_set)
    if SIGNAL_SOURCE_POS is not None:
        all_points_for_limits.append(SIGNAL_SOURCE_POS)

    if all_points_for_limits:
        all_points_np = np.array(
            [
                p
                for p in all_points_for_limits
                if p is not None and np.asarray(p).shape != ()
            ]
        )
        if all_points_np.ndim == 2 and all_points_np.shape[0] > 0:
            ax.set_xlim(
                np.min(all_points_np[:, 0]) - 1, np.max(all_points_np[:, 0]) + 1
            )
            ax.set_ylim(
                np.min(all_points_np[:, 1]) - 1, np.max(all_points_np[:, 1]) + 1
            )
        else:
            ax.set_xlim(-1, 11)
            ax.set_ylim(-1, 9)
    else:
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 9)

    return (
        colors,
        robot_plots,
        orientation_lines_plots,
        robot_trail_collections,
        robot_labels,
    )


def update_plot(robots_list, step, plots):

    (
        colors,
        robot_plots,
        orientation_lines_plots,
        robot_trail_collections,
        robot_labels,
    ) = plots

    ax.set_title(f"Robot Swarm Waypoint Patrol Simulation (Step: {step})")
    for i, robot in enumerate(robots_list):
        robot_marker_color = colors[i % len(colors)]
        marker_shape = (
            "s" if robot.patrol_cycle_completed else ("X" if robot.is_rotating else "o")
        )
        robot_plots[i].set_marker(marker_shape)
        robot_plots[i].set_color(robot_marker_color)
        robot_plots[i].set_markersize(
            10 if robot.is_rotating or robot.patrol_cycle_completed else 8
        )
        robot_plots[i].set_data([robot.position[0]], [robot.position[1]])

        head_x = robot.position[0] + ORIENTATION_LINE_LENGTH * math.cos(
            robot.orientation
        )
        head_y = robot.position[1] + ORIENTATION_LINE_LENGTH * math.sin(
            robot.orientation
        )
        orientation_lines_plots[i].set_data(
            [robot.position[0], head_x], [robot.position[1], head_y]
        )
        orientation_lines_plots[i].set_color(robot_marker_color)

        if len(robot.path_history) >= 2:
            segments = []
            segment_signal_values = []
            # Path history now stores (position_array, signal_strength)
            for j in range(1, len(robot.path_history)):
                pos_prev, _ = robot.path_history[j - 1]  # Previous position
                pos_curr, sig_curr = robot.path_history[
                    j
                ]  # Current position and signal at current
                segments.append([pos_prev, pos_curr])
                segment_signal_values.append(
                    sig_curr
                )  # Color segment based on signal at its end point

            robot_trail_collections[i].set_segments(segments)
            robot_trail_collections[i].set_array(
                np.array(segment_signal_values)
            )  # Set colors for segments
        else:
            robot_trail_collections[i].set_segments([])  # Clear if not enough history

        orientation_deg = math.degrees(robot.orientation) % 360
        state_str = (
            "Done"
            if robot.patrol_cycle_completed
            else ("Rot" if robot.is_rotating else "Mov")
        )
        robot_labels[i].set_position(
            (robot.position[0] + 0.25, robot.position[1] + 0.25)
        )
        robot_labels[i].set_text(
            f"R{i} {state_str} S:{robot.sensed_signal_strength:.0f}\nO:{orientation_deg:.0f}°"
        )
        robot_labels[i].set_color(robot_marker_color)

    fig.canvas.draw()
    fig.canvas.flush_events()


def plot_signal_heatmap(
    ax,
    simulation_data,
    map_vertices=None,
    signal_source_pos=None,
    min_signal=0,
    max_signal=1500,
):
    all_x_coords = []
    all_y_coords = []
    all_signal_strengths = []

    if not simulation_data:
        print("No simulation data to plot.")
        return

    for robot_data in simulation_data:
        path_history = robot_data.get("path_history", [])
        for position, signal in path_history:
            all_x_coords.append(position[0])
            all_y_coords.append(position[1])
            all_signal_strengths.append(signal)

    if not all_x_coords:
        print("No path history points found in the simulation data.")
        return

    if map_vertices is not None and len(map_vertices) > 0:
        map_poly = np.array(map_vertices)
        ax.plot(map_poly[:, 0], map_poly[:, 1], "k-", label="Map Boundary", alpha=0.7)

    if signal_source_pos is not None:
        ax.plot(
            signal_source_pos[0],
            signal_source_pos[1],
            "yx",
            markersize=12,
            markeredgewidth=2,
            label="Signal Source",
        )

    cmap = plt.cm.viridis
    norm = Normalize(vmin=min_signal, vmax=max_signal)

    scatter = ax.scatter(
        all_x_coords,
        all_y_coords,
        c=all_signal_strengths,
        cmap=cmap,
        norm=norm,
        s=15,
        alpha=0.7,
        edgecolors="none",
    )

    cbar = plt.colorbar(scatter, label="Sensed Signal Strength")

    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.set_title("Signal Strength Heatmap from Robot Paths")
    ax.axis("equal")
    ax.grid(True, linestyle="--", alpha=0.5)

    if (map_vertices is not None and len(map_vertices) > 0) or (
        signal_source_pos is not None
    ):
        ax.legend(loc="upper right")


def run_simulation(cell_id, plots, waypoints):
    robots = initialize_robots(waypoints)
    simulation_data = []
    source_info = {"pos": SIGNAL_SOURCE_POS, "params": EMITTER_PARAMS}

    for step in range(SIMULATION_STEPS):
        all_robots_finished_patrol = True
        for robot in robots:
            if not robot.patrol_cycle_completed:
                all_robots_finished_patrol = False

            robot.sense_signal(source_info["pos"], EMITTER_PARAMS)
            robot.move()

        update_plot(robots, step + 1, plots)

        if all_robots_finished_patrol:
            for robot in robots:
                if robot.patrol_cycle_completed and (
                    not robot.path_history
                    or not np.array_equal(robot.path_history[-1][0], robot.position)
                ):
                    robot.path_history.append(
                        (robot.position.copy(), robot.sensed_signal_strength)
                    )
                    if (
                        robot.max_path_history > 0
                        and len(robot.path_history) > robot.max_path_history
                    ):
                        robot.path_history.pop(0)
            break

        time.sleep(0.05)

    if not all_robots_finished_patrol:
        for robot in robots:
            if not robot.patrol_cycle_completed and (
                not robot.path_history
                or not np.array_equal(robot.path_history[-1][0], robot.position)
            ):
                robot.path_history.append(
                    (robot.position.copy(), robot.sensed_signal_strength)
                )
                if (
                    robot.max_path_history > 0
                    and len(robot.path_history) > robot.max_path_history
                ):
                    robot.path_history.pop(0)

    for robot in robots:
        simulation_data.append(
            {
                "cell_id": cell_id,
                "robot_id": robot.robot_id,
                "path_history": list(robot.path_history),
            }
        )

    plt.ioff()
    plt.close(fig)
    return simulation_data


def plot_orthogonal_line(
    ax, max_point, edge_start, edge_end, length=0.5, lw=2, color="cyan"
):
    x0, y0 = max_point
    x1, y1 = edge_start
    x2, y2 = edge_end

    if x2 == x1:
        p1 = (x0 - length / 2, y0)
        p2 = (x0 + length / 2, y0)
    else:
        m = (y2 - y1) / (x2 - x1)
        if m == 0:
            p1 = (x0, y0 - length / 2)
            p2 = (x0, y0 + length / 2)
        else:
            m_perp = -1 / m
            dx = 1 / np.sqrt(1 + m_perp**2)
            dy = m_perp * dx
            p1 = (x0 - dx * length / 2, y0 - dy * length / 2)
            p2 = (x0 + dx * length / 2, y0 + dy * length / 2)

    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw)


if __name__ == "__main__":
    cell_data = []
    patrol_data = {}

    map_polygon = Polygon(MAP_VERTICES)
    cells, xbound, ybound = voronoi_cells(map_polygon)

    for i, cell in enumerate(cells):

        print(f"Cell {i}")

        params, patrols = area_coverage_optimiser(cell, xbound, ybound)

        # plot_area_coverage(cell, patrols)

        robot_waypoints = []
        for patrol in patrols:
            robot_waypoints.append(np.array(patrol.patrol.exterior.coords))

        patrol_data[f"cell_{i}"] = robot_waypoints

        plt.ion()
        fig, ax = plt.subplots(figsize=(12, 8))
        initialize_plot(ax, cell)
        plots = add_robots_to_plot(ax)

        robot_data = run_simulation(i, plots, robot_waypoints)
        cell_data.append(robot_data)  # <-- Append inside the loop

    # Combine all simulation data from all cells for a single heatmap
    all_simulation_data = [entry for cell in cell_data for entry in cell]

    fig, ax = plt.subplots()
    plot_signal_heatmap(
        ax,
        all_simulation_data,
        map_vertices=MAP_VERTICES,
        signal_source_pos=SIGNAL_SOURCE_POS,
        min_signal=MIN_SIGNAL_FOR_COLORMAP,
        max_signal=MAX_SIGNAL_FOR_COLORMAP,
    )
    plt.show()

    with open("patrol_data.csv", "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cell_id", "robot_id", "x", "y"])

        for cell_id_str, robots in patrol_data.items():
            cell_id = int(cell_id_str.split("_")[1])
            for robot_id, waypoints in enumerate(robots):
                for x, y in waypoints:
                    writer.writerow([cell_id, robot_id, x, y])

    with open(CSV_FILE, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["cell_id", "robot_id", "x", "y", "strength"])  # header

        for entry in all_simulation_data:
            cell_id = entry["cell_id"]
            robot_id = entry["robot_id"]
            for pos, strength in entry["path_history"]:
                writer.writerow([cell_id, robot_id, pos[0], pos[1], strength])

    df = pd.read_csv("simulation_data.csv")

    robot_data = {
        robot_id: group[["x", "y", "strength"]]
        for robot_id, group in df.groupby("robot_id")
    }

    fig, ax = plt.subplots()
    ax.plot(MAP_VERTICES[:, 0], MAP_VERTICES[:, 1])

    for cell_id, cell_group in df.groupby("cell_id"):
        for robot_id, robot in cell_group.groupby("robot_id"):

            ax.scatter(robot["x"], robot["y"], c=robot["strength"], cmap="viridis")

            vertices = patrol_data[f"cell_{cell_id}"][
                robot_id
            ]  # Assumes robot_vertices is indexed by robot_id

            for i in range(len(vertices) - 1):  # closed loop assumed
                start_x, start_y = vertices[i]
                end_x, end_y = vertices[i + 1]

                # Match start and end vertices
                start_match = robot[
                    np.isclose(robot["x"], start_x) & np.isclose(robot["y"], start_y)
                ]
                end_match = robot[
                    np.isclose(robot["x"], end_x) & np.isclose(robot["y"], end_y)
                ]

                if not start_match.empty and not end_match.empty:

                    if i == 0:
                        start_idx = start_match.index[0]
                    else:
                        start_idx = start_match.index[-1]
                    end_idx = end_match.index[-1]

                    i1, i2 = sorted([start_idx, end_idx])
                    df_split = robot.loc[i1:i2]

                    max_strength_row = df_split.loc[df_split["strength"].idxmax()]

                    # Skip if max point is at vertex
                    if not (
                        (
                            np.isclose(max_strength_row["x"], start_x, atol=1e-3)
                            and np.isclose(max_strength_row["y"], start_y, atol=1e-3)
                        )
                        or (
                            np.isclose(max_strength_row["x"], end_x)
                            and np.isclose(max_strength_row["y"], end_y)
                        )
                    ):
                        ax.scatter(
                            max_strength_row["x"],
                            max_strength_row["y"],
                            marker="x",
                            color="r",
                            s=10,
                        )

                        # Calculate gradient of segment
                        dx = end_x - start_x
                        dy = end_y - start_y
                        gradient = np.inf if np.isclose(dx, 0.0) else dy / dx

                        # Perpendicular gradient
                        if np.isclose(gradient, 0.0):
                            perp_grad = np.inf
                        elif np.isinf(gradient):
                            perp_grad = 0.0
                        else:
                            perp_grad = -1 / gradient

                        # Line through max point with gradient = perp_grad
                        length = 50.0  # adjustable
                        x0, y0 = max_strength_row["x"], max_strength_row["y"]

                        if np.isinf(perp_grad):
                            x_vals = [x0, x0]
                            y_vals = [y0 - length / 2, y0 + length / 2]
                        else:
                            dx_line = length / 2 / np.sqrt(1 + perp_grad**2)
                            x_vals = [x0 - dx_line, x0 + dx_line]
                            y_vals = [perp_grad * (x - x0) + y0 for x in x_vals]

                        ax.plot(x_vals, y_vals, "r--")

                else:
                    print(
                        f"Segment skipped: no match for robot {robot_id} in cell {cell_id}, segment {i}"
                    )
                    
    ax.scatter(
        SIGNAL_SOURCE_POS[0], SIGNAL_SOURCE_POS[1], marker="x", color="blue", s=50
    )

    ax.axis("equal")
    plt.show()
