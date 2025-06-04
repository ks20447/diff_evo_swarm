import time
import math
import numpy as np
from Robot import Robot
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
        line_length=0.5,
    ):

        self.sim_steps = params["sim_steps"]
        self.map_vertices = params["map_vertices"]
        self.num_robots = params["num_robots"]
        self.source = params["source"]

        self.max_path_history_length = 0
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

    def initialize_plot(self, cell):

        self.fig, self.ax = plt.subplots()

        self.ax.plot(
            self.map_vertices[:, 0], self.map_vertices[:, 1], "k-", label="Map Boundary"
        )
        self.ax.plot(
            self.source["pos"][0],
            self.source["pos"][1],
            "yx",
            markersize=10,
            markeredgewidth=1.5,
            label="Signal Source",
        )
        self.ax.plot(cell.exterior.xy[0], cell.exterior.xy[1], color="r")

    def initialise(self):

        self.robots = []
        for i in range(self.num_robots):

            waypoints = self.waypoints[i]
            initial_pos = waypoints[0] if len(waypoints) > 0 else None

            self.robots.append(
                Robot(
                    robot_id=i,
                    waypoints=waypoints,
                    source=self.source,
                    initial_pos=initial_pos,
                    max_history=self.max_path_history_length,
                )
            )

    def add_robots_to_plot(self):

        robot_plots = []
        waypoint_plots_markers = []
        waypoint_plots_lines = []
        robot_labels = []
        orientation_lines_plots = []
        robot_trail_collections = []
        colors = ["r", "g", "b", "c", "m", "y", "orange"]
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
            orientation_lines_plots.append(orientation_line)

            trail_collection = LineCollection([], cmap=self.cmap, norm=norm)
            trail_collection.set_linewidth(3)
            self.ax.add_collection(trail_collection)
            robot_trail_collections.append(trail_collection)

            if i < len(self.waypoints) and self.waypoints[i].size > 0:
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
                0, 0, "", fontsize=8, color=robot_marker_color, va="bottom"
            )
            robot_labels.append(label)

        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")
        self.ax.set_title("Robot Swarm Waypoint Patrol Simulation")
        handles, legend_labels_list = self.ax.get_legend_handles_labels()
        unique_labels_dict = {}
        for handle, label_text in zip(handles, legend_labels_list):
            if label_text not in unique_labels_dict:
                unique_labels_dict[label_text] = handle
        self.ax.legend(
            unique_labels_dict.values(),
            unique_labels_dict.keys(),
            loc="upper left",
            fontsize="small",
        )

        self.ax.axis("equal")
        self.ax.grid(True)

        if self.num_robots > 0:
            sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=norm)
            sm.set_array([])
            _ = self.fig.colorbar(
                sm,
                ax=self.ax,
                label="Signal Strength",
                orientation="vertical",
                pad=0.02,
                aspect=30,
            )

        all_points_for_limits = list(self.map_vertices)
        for wps_set in self.waypoints:
            if wps_set.size > 0:
                all_points_for_limits.extend(wps_set)
        if self.source["pos"] is not None:
            all_points_for_limits.append(self.source["pos"])

        if all_points_for_limits:
            all_points_np = np.array(
                [
                    p
                    for p in all_points_for_limits
                    if p is not None and np.asarray(p).shape != ()
                ]
            )
            if all_points_np.ndim == 2 and all_points_np.shape[0] > 0:
                self.ax.set_xlim(
                    np.min(all_points_np[:, 0]) - 1, np.max(all_points_np[:, 0]) + 1
                )
                self.ax.set_ylim(
                    np.min(all_points_np[:, 1]) - 1, np.max(all_points_np[:, 1]) + 1
                )
            else:
                self.ax.set_xlim(-1, 11)
                self.ax.set_ylim(-1, 9)
        else:
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

        self.all_robots_finished_patrol = True

        for robot in self.robots:
            if not robot.patrol_cycle_completed:
                self.all_robots_finished_patrol = False

            robot.sense_signal(self.source["pos"], self.source["params"])
            robot.move()

        if self.plotting:
            self.update_plot()

        if self.all_robots_finished_patrol:
            for robot in self.robots:
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
            return False

        time.sleep(self.sleep_time)

        return True

    def update_plot(self):

        colors, robot_plots, orientation_lines_plots, robot_trail_collections, robot_labels = self.plots

        self.ax.set_title(f"Robot Swarm Waypoint Patrol Simulation (Step: {self.step_count})")
        
        for i, robot in enumerate(self.robots):
            robot_marker_color = colors[i % len(colors)]
            marker_shape = (
                "s"
                if robot.patrol_cycle_completed
                else ("X" if robot.is_rotating else "o")
            )
            robot_plots[i].set_marker(marker_shape)
            robot_plots[i].set_color(robot_marker_color)
            robot_plots[i].set_markersize(
                10 if robot.is_rotating or robot.patrol_cycle_completed else 8
            )
            robot_plots[i].set_data([robot.position[0]], [robot.position[1]])

            head_x = robot.position[0] + self.line_length * math.cos(
                robot.orientation
            )
            head_y = robot.position[1] + self.line_length * math.sin(
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
                robot_trail_collections[i].set_segments(
                    []
                )  # Clear if not enough history

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

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


    def run(self, cell, robot_waypoints):

        print("Sim Started")

        self.waypoints = robot_waypoints

        cell_id = cell["cell_id"]
        cell_polygon = cell["cell_polygon"]

        self.initialise()

        if self.plotting:
            self.initialize_plot(cell_polygon)
            self.add_robots_to_plot()

        for _ in range(self.sim_steps):

            self.step_count += 1
            cont = self.step()

            if not cont:
                break

        if not self.all_robots_finished_patrol:
            for robot in self.robots:
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

        for robot in self.robots:
            self.sim_data.append(
                {
                    "cell_id": cell_id,
                    "robot_id": robot.robot_id,
                    "path_history": list(robot.path_history),
                }
            )

        print("Sim Ended")

        if self.plotting:
            plt.ioff()
            plt.close(self.fig)
            
        
        return self.sim_data
