import time
import math
import numpy as np

# Assuming Patrol classes are defined elsewhere if used, or remove if not.
# from Patrol import OmniPatrol, DirPatrol
from Robot import Robot  # Make sure this is the updated Robot.py
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection


class Simulator:

    def __init__(
        self,
        params,
        plotting=False,
        sleep_time=0,
        cmap=plt.cm.viridis,
        min_cmap=-50.0,
        max_cmap=20.0,
        line_length=1.0,
    ):

        self.sim_steps = params["sim_steps"]
        self.map_vertices = params["map_vertices"]
        self.num_robots = params["num_robots"]
        self.source = params["source"]
        self.sense_type = params["sense_type"]  # Used by Robot class

        self.max_path_history_length = 0  # Passed to Robot, 0 for unlimited
        self.all_robots_finished_patrol = False

        self.sim_data = []
        self.step_count = 0
        self.sleep_time = sleep_time

        self.plotting = plotting

        if self.plotting:
            self.cmap = cmap
            self.min_cmap = min_cmap
            self.max_cmap = max_cmap
            self.line_length = line_length

    def initialize_plot(self, cell):  # Assuming 'cell' is a shapely geometry or similar

        self.fig, self.ax = plt.subplots()

        self.ax.plot(
            self.map_vertices[:, 0], self.map_vertices[:, 1], "k-", label="Map Boundary"
        )
        if self.source and "pos" in self.source and self.source["pos"] is not None:
            self.ax.plot(
                self.source["pos"][0],
                self.source["pos"][1],
                "yx",
                markersize=10,
                markeredgewidth=1.5,
                label="Signal Source",
            )
        if hasattr(cell, "exterior"):  # Check if cell has an exterior (like a polygon)
            self.ax.plot(
                cell.exterior.xy[0],
                cell.exterior.xy[1],
                color="r",
                label="Cell Boundary",
            )

    def initialise(self):

        self.robots = []
        for i in range(self.num_robots):

            # Ensure self.waypoints is populated before calling initialise, e.g., in run()
            if i < len(self.waypoints):
                waypoints_for_robot = self.waypoints[i]
            else:
                waypoints_for_robot = np.array(
                    []
                )  # Default to empty if not enough waypoints defined

            initial_pos = (
                waypoints_for_robot[0] if len(waypoints_for_robot) > 0 else None
            )

            self.robots.append(
                Robot(
                    robot_id=i,
                    waypoints=waypoints_for_robot,
                    source=self.source,
                    initial_pos=initial_pos,
                    max_history=self.max_path_history_length,  # Robot handles 0 as unlimited
                    sense_type=self.sense_type,
                )
            )

    def add_robots_to_plot(self):

        robot_plots = []
        waypoint_plots_markers = []
        waypoint_plots_lines = []
        robot_labels = []
        orientation_lines_plots = []
        robot_trail_collections = []
        # Ensure self.waypoints is accessible and populated
        if not hasattr(self, "waypoints") or self.waypoints is None:
            # Initialize waypoints as an empty list or handle error if it should be populated by now
            self.waypoints = [np.array([]) for _ in range(self.num_robots)]

        colors = [
            "r",
            "g",
            "b",
            "c",
            "m",
            "y",
            "orange",
            "purple",
            "brown",
            "pink",
        ]  # Expanded colors
        norm = Normalize(vmin=self.min_cmap, vmax=self.max_cmap)

        for i in range(self.num_robots):
            robot_marker_color = colors[i % len(colors)]
            (plot,) = self.ax.plot(
                [],
                [],
                marker="o",
                color=robot_marker_color,
                markersize=8,
                label=f"Robot {i}",
            )
            robot_plots.append(plot)

            (orientation_line,) = self.ax.plot(
                [], [], linestyle="-", color=robot_marker_color, linewidth=2
            )
            (antenna_line_a,) = self.ax.plot(
                [], [], linestyle="-", color=robot_marker_color, linewidth=2
            )
            (antenna_line_b,) = self.ax.plot(
                [], [], linestyle="-", color=robot_marker_color, linewidth=2
            )
            orientation_lines_plots.append(
                [orientation_line, antenna_line_a, antenna_line_b]
            )

            trail_collection = LineCollection([], cmap=self.cmap, norm=norm)
            trail_collection.set_linewidth(3)
            self.ax.add_collection(trail_collection)
            robot_trail_collections.append(trail_collection)

            # Ensure there are waypoints defined for the robot before trying to plot them
            if (
                i < len(self.waypoints)
                and self.waypoints[i] is not None
                and self.waypoints[i].size > 0
            ):
                wps = self.waypoints[i]
                (wp_line,) = self.ax.plot(
                    wps[:, 0],
                    wps[:, 1],
                    linestyle="--",
                    color=robot_marker_color,
                    alpha=0.3,
                )
                waypoint_plots_lines.append(wp_line)
                (wp_marker,) = self.ax.plot(
                    wps[:, 0],
                    wps[:, 1],
                    marker="x",
                    color=robot_marker_color,
                    markersize=7,
                    alpha=0.7,
                    linestyle="None",
                )
                waypoint_plots_markers.append(wp_marker)

            label = self.ax.text(
                0, 0, "", fontsize=8, color=robot_marker_color, va="bottom", ha="left"
            )
            robot_labels.append(label)

        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title("Robot Swarm Waypoint Patrol Simulation")
        handles, legend_labels_list = self.ax.get_legend_handles_labels()
        unique_labels_dict = {}
        for handle, label_text in zip(handles, legend_labels_list):
            if label_text not in unique_labels_dict:  # Keep only unique labels
                unique_labels_dict[label_text] = handle
        self.ax.legend(
            unique_labels_dict.values(),
            unique_labels_dict.keys(),
            loc="upper left",
            fontsize="small",
        )

        self.ax.axis("equal")
        self.ax.grid(True)

        if self.num_robots > 0:  # Add colorbar only if there are robots
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            sm.set_array([])  # Empty array for scalar mappable
            self.fig.colorbar(
                sm,
                ax=self.ax,
                label="Signal Strength (Displayed)",
                orientation="vertical",
                pad=0.02,  # Adjust padding
                aspect=30,  # Adjust aspect ratio for thickness
            )

        # Consolidate points for auto-limits
        all_points_for_limits = (
            list(self.map_vertices.reshape(-1, 2)) if self.map_vertices.size > 0 else []
        )
        for wps_set in self.waypoints:
            if wps_set is not None and wps_set.size > 0:
                all_points_for_limits.extend(wps_set.reshape(-1, 2))
        if self.source and "pos" in self.source and self.source["pos"] is not None:
            all_points_for_limits.append(self.source["pos"])

        if all_points_for_limits:
            all_points_np = np.array(all_points_for_limits)
            if all_points_np.ndim == 2 and all_points_np.shape[0] > 0:
                x_min, x_max = np.min(all_points_np[:, 0]), np.max(all_points_np[:, 0])
                y_min, y_max = np.min(all_points_np[:, 1]), np.max(all_points_np[:, 1])
                x_padding = max(
                    1.0, (x_max - x_min) * 0.1
                )  # Add 10% padding or at least 1 unit
                y_padding = max(1.0, (y_max - y_min) * 0.1)
                self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
                self.ax.set_ylim(y_min - y_padding, y_max + y_padding)
            else:  # Fallback if points are not as expected
                self.ax.set_xlim(-1, 11)
                self.ax.set_ylim(-1, 9)
        else:  # Default limits if no points
            self.ax.set_xlim(-1, 11)
            self.ax.set_ylim(-1, 9)

        self.plots = (
            colors,
            robot_plots,
            orientation_lines_plots,
            robot_trail_collections,
            robot_labels,
        )

    def step(self):
        # Determine if all robots have finished *after* their move for this step
        for robot in self.robots:
            robot.sense_signal(self.source["pos"], self.source["params"])
            robot.move()  # Robot.move() now handles its own path_history appending

        # Check completion status after all robots have moved
        self.all_robots_finished_patrol = True
        for robot in self.robots:
            if not robot.patrol_cycle_completed or robot.is_rotating:
                self.all_robots_finished_patrol = False
                break

        if self.plotting:
            self.update_plot()

        if self.all_robots_finished_patrol:
            # Robots manage their own final history append upon completion.
            # No explicit append needed here.
            return False  # Signal to stop simulation

        if self.sleep_time > 0:
            time.sleep(self.sleep_time)

        return True  # Signal to continue simulation

    def update_plot(self):
        (
            colors,
            robot_plots,
            orientation_lines_plots,
            robot_trail_collections,
            robot_labels,
        ) = self.plots

        self.ax.set_title(
            f"Robot Swarm Waypoint Patrol Simulation (Step: {self.step_count})"
        )

        for i, robot in enumerate(self.robots):
            robot_marker_color = colors[i % len(colors)]
            marker_shape = (
                "s"  # Square for completed
                if robot.patrol_cycle_completed
                else ("X" if robot.is_rotating else "o")  # X for rotating, o for moving
            )
            robot_plots[i].set_marker(marker_shape)
            robot_plots[i].set_color(robot_marker_color)
            robot_plots[i].set_markersize(
                10 if robot.is_rotating or robot.patrol_cycle_completed else 8
            )
            robot_plots[i].set_data([robot.position[0]], [robot.position[1]])

            # Plot orientation line
            head_x = robot.position[0] + self.line_length * math.cos(robot.orientation)
            head_y = robot.position[1] + self.line_length * math.sin(robot.orientation)

            orientation_lines_plots[i][0].set_data(
                [robot.position[0], head_x], [robot.position[1], head_y]
            )
            orientation_lines_plots[i][0].set_color(robot_marker_color)

            for j, offset in enumerate(robot.offsets):
                head_x = robot.position[0] + self.line_length * math.cos(
                    robot.orientation + offset
                )
                head_y = robot.position[1] + self.line_length * math.sin(
                    robot.orientation + offset
                )

                orientation_lines_plots[i][j + 1].set_data(
                    [robot.position[0], head_x], [robot.position[1], head_y]
                )
                orientation_lines_plots[i][j + 1].set_color(robot_marker_color)

            # Plot robot trail
            if len(robot.path_history) >= 2:
                segments = []
                segment_signal_values = []
                for j in range(1, len(robot.path_history)):
                    pos_prev, _, _ = robot.path_history[j - 1]
                    pos_curr, _, sig_curr_raw = robot.path_history[
                        j
                    ]  # sig_curr_raw can be scalar or array

                    signal_for_color = self.min_cmap  # Default color value
                    if isinstance(sig_curr_raw, np.ndarray) and sig_curr_raw.size > 0:
                        signal_for_color = sig_curr_raw[
                            0
                        ]  # Use first element for coloring
                    elif isinstance(
                        sig_curr_raw, (float, int, np.number)
                    ):  # Check for scalar numeric types
                        signal_for_color = sig_curr_raw

                    segments.append([pos_prev, pos_curr])
                    segment_signal_values.append(signal_for_color)

                robot_trail_collections[i].set_segments(segments)
                # Ensure array is float for color mapping
                robot_trail_collections[i].set_array(
                    np.array(segment_signal_values, dtype=float)
                )
            else:
                robot_trail_collections[i].set_segments(
                    []
                )  # Clear if not enough history

            # Update robot label
            orientation_deg = math.degrees(robot.orientation) % 360
            state_str = (
                "Done"
                if robot.patrol_cycle_completed
                else ("Rot" if robot.is_rotating else "Mov")
            )

            current_signal_raw = robot.sensed_signal_strength
            signal_display_val = 0.0  # Default display value
            if (
                isinstance(current_signal_raw, np.ndarray)
                and current_signal_raw.size > 0
            ):
                signal_display_val = current_signal_raw[1]  # Display the first element
            elif isinstance(
                current_signal_raw, (float, int, np.number)
            ):  # Check for scalar numeric types
                signal_display_val = current_signal_raw

            robot_labels[i].set_position(
                (robot.position[0] + 0.25, robot.position[1] + 0.25)
            )
            robot_labels[i].set_text(
                f"R{i} {state_str} S:{signal_display_val:.0f}\nO:{orientation_deg:.0f}°"
            )
            robot_labels[i].set_color(robot_marker_color)

        self.fig.canvas.draw_idle()  # Use draw_idle for better performance with some backends
        self.fig.canvas.flush_events()

    def run(self, cell, robot_waypoints):
        print("Sim Started")

        self.waypoints = robot_waypoints  # Store waypoints for use in initialise and add_robots_to_plot

        # Ensure cell is a dictionary with 'cell_id' and 'cell_polygon'
        cell_id = cell.get("cell_id", "unknown_cell")
        cell_polygon = cell.get(
            "cell_polygon", None
        )  # This should be a geometry object for initialize_plot

        self.initialise()  # Initialise robots with their waypoints

        if self.plotting:
            if cell_polygon is None:
                print(
                    "Warning: cell_polygon not provided, skipping cell boundary plot."
                )
                # Create a dummy polygon or handle as per requirements if plotting is essential
                from shapely.geometry import Polygon as ShapelyPolygon

                cell_polygon = ShapelyPolygon()  # Empty polygon if not available
            self.initialize_plot(cell_polygon)
            self.add_robots_to_plot()

        for step_num in range(self.sim_steps):
            self.step_count = step_num + 1  # Update step_count for display

            continue_sim = self.step()
            if not continue_sim:
                print(f"All robots completed patrol at step {self.step_count}.")
                break

        if self.step_count == self.sim_steps and not self.all_robots_finished_patrol:
            print(
                f"Simulation ended after {self.sim_steps} steps; not all robots finished."
            )

        # Data collection - Robot.path_history is managed by the Robot class itself
        # Clear previous sim_data for this run
        self.sim_data = []
        for robot in self.robots:
            self.sim_data.append(
                {
                    "cell_id": cell_id,
                    "robot_id": robot.robot_id,
                    "path_history": [
                        (
                            pos.tolist(),
                            ori,
                            sig.tolist() if isinstance(sig, np.ndarray) else sig,
                        )
                        for pos, ori, sig in robot.path_history
                    ],
                }
            )

        print("Sim Ended")

        if self.plotting:
            # plt.ioff() # Not typically needed here unless ion was used explicitly and needs to be turned off
            if hasattr(self, "fig") and self.fig is not None:
                plt.close(self.fig)  # Close the figure to free memory

        return self.sim_data, self.step_count

