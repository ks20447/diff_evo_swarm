import math
import time
from Robot import Robot
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from Patrol import OmniPatrol
from scipy.optimize import differential_evolution
from shapely.geometry import Polygon
from optimize import generate_bounds, generalized_objective
from Voronoi import create_voronoi_partitions_on_map, plot_results

# --- Configuration ---
SIMULATION_STEPS = 250
ORIENTATION_LINE_LENGTH = 0.3

SIGNAL_SOURCE_POS = np.array([5.0, 5.0])
SIGNAL_POWER = 1000.0

# Trail Color Configuration
MIN_SIGNAL_FOR_COLORMAP = 0.0
MAX_SIGNAL_FOR_COLORMAP = 100.0  # Adjusted based on typical values, might need tuning
SELECTED_CMAP = plt.cm.viridis
MAX_PATH_HISTORY_LENGTH = (
    0  # Number of past positions to show in the trail, 0 for infinite
)

scale = 2
shift = (-5, -4)
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
    * scale
    + shift
)

map_polygon = Polygon(MAP_VERTICES)

minx, miny, maxx, maxy = map_polygon.bounds
x_coords = np.linspace(minx, maxx, 6)
y_coords = np.linspace(miny, maxy, 7)
grid_points_input = []
for x in x_coords:
    for y in y_coords:
        grid_points_input.append((x, y))

clipped_cells, used_sites = create_voronoi_partitions_on_map(
    map_polygon, grid_points_input
)

plot_results(map_polygon, clipped_cells, used_sites)

cell = clipped_cells[0]
MAP_VERTICES = np.array(cell.exterior.coords)

shape_config = {"Triangle": 3, "Square": 0, "Hexagon": 0}
S_R = 4.0
weights = [1.0, 100.0, 0, 0]  # Weights for the objective function

bounds = generate_bounds(shape_config, S_R, (minx, maxx), (miny, maxy))

result = differential_evolution(
    generalized_objective,
    bounds,
    args=(S_R, weights, shape_config, cell),
    maxiter=500,
    polish=True,
    init="halton",
    updating="deferred",
    workers=-1,
)

params = result.x
ROBOT_WAYPOINTS = []
patrols = []

idx = 0
for shape, count in shape_config.items():
    for _ in range(count):
        x = params[idx]
        y = params[idx + 1]
        angle = params[idx + 2]
        length = params[idx + 3]

        patrol = OmniPatrol(shape, (x, y), length, S_R, angle=angle)
        patrols.append(patrol)
        ROBOT_WAYPOINTS.append(np.array(patrol.patrol.exterior.coords))

        idx += 4


NUM_ROBOTS = sum(shape_config.values())

fig, ax = plt.subplots()

x, y = cell.exterior.xy
ax.axis("equal")
ax.plot(x, y)

for patrol in patrols:

    patrol.plot(ax, outline=True)

plt.show()


# --- Simulation Setup ---
def initialize_robots():
    robots = []
    source_info = {"pos": SIGNAL_SOURCE_POS, "strength": SIGNAL_POWER}
    for i in range(NUM_ROBOTS):
        if i < len(ROBOT_WAYPOINTS):
            waypoints = ROBOT_WAYPOINTS[i]
            # initial_pos defaults to waypoints[0] in Robot class if None
            initial_pos = waypoints[0] if len(waypoints) > 0 else None 
        else:
            print(f"Warning: Not enough waypoint sets defined. Using default for Robot {i}")
            waypoints = np.array([[float(i+1), float(i+1)], [float(i+2), float(i+2)]])
            initial_pos = waypoints[0]
        
        # Pass source_info and MAX_PATH_HISTORY_LENGTH to the Robot constructor
        robots.append(Robot(robot_id=i, 
                            waypoints=waypoints, 
                            source=source_info, 
                            initial_pos=initial_pos,
                            max_history=MAX_PATH_HISTORY_LENGTH
                            # speed can be passed here to override ROBOT_SPEED from robot_class.py
                           ))
    return robots

# --- Visualization ---
plt.ion() 
fig, ax = plt.subplots(figsize=(12, 8))

map_plot, = ax.plot(MAP_VERTICES[:, 0], MAP_VERTICES[:, 1], 'k-', label="Map Boundary")
signal_source_plot, = ax.plot(SIGNAL_SOURCE_POS[0], SIGNAL_SOURCE_POS[1], 'yx', markersize=10, markeredgewidth=1.5, label="Signal Source")

robot_plots = []
waypoint_plots_markers = []
waypoint_plots_lines = []
robot_labels = []
orientation_lines_plots = [] 
robot_trail_collections = []
colors = ['r', 'g', 'b', 'c', 'm', 'y', 'orange'] 
norm = Normalize(vmin=MIN_SIGNAL_FOR_COLORMAP, vmax=MAX_SIGNAL_FOR_COLORMAP)

for i in range(NUM_ROBOTS):
    robot_marker_color = colors[i % len(colors)]
    plot, = ax.plot([], [], marker='o', color=robot_marker_color, markersize=8, label=f'Robot {i}') 
    robot_plots.append(plot)
    
    orientation_line, = ax.plot([], [], linestyle='-', color=robot_marker_color, linewidth=2) 
    orientation_lines_plots.append(orientation_line)

    trail_collection = LineCollection([], cmap=SELECTED_CMAP, norm=norm)
    trail_collection.set_linewidth(3)
    ax.add_collection(trail_collection)
    robot_trail_collections.append(trail_collection)

    if i < len(ROBOT_WAYPOINTS) and ROBOT_WAYPOINTS[i].size > 0:
        wps = ROBOT_WAYPOINTS[i]
        wp_line, = ax.plot(wps[:, 0], wps[:, 1], linestyle='--', color=robot_marker_color, alpha=0.3)
        waypoint_plots_lines.append(wp_line)
        wp_marker, = ax.plot(wps[:, 0], wps[:, 1], marker='x', color=robot_marker_color, markersize=7, alpha=0.7, linestyle='None')
        waypoint_plots_markers.append(wp_marker)
    
    label = ax.text(0, 0, '', fontsize=8, color=robot_marker_color, va='bottom') 
    robot_labels.append(label)

ax.set_xlabel("X Coordinate")
ax.set_ylabel("Y Coordinate")
ax.set_title("Robot Swarm Waypoint Patrol Simulation")
handles, legend_labels_list = ax.get_legend_handles_labels()
unique_labels_dict = {}
for handle, label_text in zip(handles, legend_labels_list):
    if label_text not in unique_labels_dict: unique_labels_dict[label_text] = handle
ax.legend(unique_labels_dict.values(), unique_labels_dict.keys(), loc='upper left', fontsize='small')

ax.axis('equal')
ax.grid(True)

if NUM_ROBOTS > 0:
    sm = plt.cm.ScalarMappable(cmap=SELECTED_CMAP, norm=norm)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=ax, label="Signal Strength", orientation="vertical", pad=0.02, aspect=30)

all_points_for_limits = list(MAP_VERTICES)
for wps_set in ROBOT_WAYPOINTS:
    if wps_set.size > 0: all_points_for_limits.extend(wps_set)
if SIGNAL_SOURCE_POS is not None: all_points_for_limits.append(SIGNAL_SOURCE_POS)

if all_points_for_limits:
    all_points_np = np.array([p for p in all_points_for_limits if p is not None and np.asarray(p).shape != ()])
    if all_points_np.ndim == 2 and all_points_np.shape[0] > 0:
        ax.set_xlim(np.min(all_points_np[:,0]) - 1, np.max(all_points_np[:,0]) + 1)
        ax.set_ylim(np.min(all_points_np[:,1]) - 1, np.max(all_points_np[:,1]) + 1)
    else: ax.set_xlim(-1, 11); ax.set_ylim(-1, 9)
else: ax.set_xlim(-1, 11); ax.set_ylim(-1, 9)


def update_plot(robots_list, step):
    ax.set_title(f"Robot Swarm Waypoint Patrol Simulation (Step: {step})")
    for i, robot in enumerate(robots_list):
        robot_marker_color = colors[i % len(colors)]
        marker_shape = 's' if robot.patrol_cycle_completed else ('X' if robot.is_rotating else 'o')
        robot_plots[i].set_marker(marker_shape)
        robot_plots[i].set_color(robot_marker_color)
        robot_plots[i].set_markersize(10 if robot.is_rotating or robot.patrol_cycle_completed else 8)
        robot_plots[i].set_data([robot.position[0]], [robot.position[1]])
        
        head_x = robot.position[0] + ORIENTATION_LINE_LENGTH * math.cos(robot.orientation)
        head_y = robot.position[1] + ORIENTATION_LINE_LENGTH * math.sin(robot.orientation)
        orientation_lines_plots[i].set_data([robot.position[0], head_x], [robot.position[1], head_y])
        orientation_lines_plots[i].set_color(robot_marker_color)

        if len(robot.path_history) >= 2:
            segments = []
            segment_signal_values = []
            # Path history now stores (position_array, signal_strength)
            for j in range(1, len(robot.path_history)):
                pos_prev, _ = robot.path_history[j-1] # Previous position
                pos_curr, sig_curr = robot.path_history[j] # Current position and signal at current
                segments.append([pos_prev, pos_curr])
                segment_signal_values.append(sig_curr) # Color segment based on signal at its end point
            
            robot_trail_collections[i].set_segments(segments)
            robot_trail_collections[i].set_array(np.array(segment_signal_values)) # Set colors for segments
        else:
            robot_trail_collections[i].set_segments([]) # Clear if not enough history

        orientation_deg = math.degrees(robot.orientation) % 360
        state_str = "Done" if robot.patrol_cycle_completed else ("Rot" if robot.is_rotating else "Mov")
        robot_labels[i].set_position((robot.position[0] + 0.25, robot.position[1] + 0.25))
        robot_labels[i].set_text(f'R{i} {state_str} S:{robot.sensed_signal_strength:.0f}\nO:{orientation_deg:.0f}°')
        robot_labels[i].set_color(robot_marker_color)

    fig.canvas.draw()
    fig.canvas.flush_events()

# --- Main Simulation Loop ---
def run_simulation():
    robots = initialize_robots()
    simulation_data = [] 
    source_info = {"pos": SIGNAL_SOURCE_POS, "strength": SIGNAL_POWER}


    for step in range(SIMULATION_STEPS):
        all_robots_finished_patrol = True
        for robot in robots:
            if not robot.patrol_cycle_completed:
                all_robots_finished_patrol = False
            
            robot.move() 
            # The Robot class's move method now handles path_history appends internally.
            # Signal sensing is also done within the robot's methods or init.
            # We need to call sense_signal here to update the strength for the *current* step
            # before it's potentially appended to path_history by the *next* move() call or by visualization.
            robot.sense_signal(source_info["pos"], source_info["strength"])
            
        update_plot(robots, step + 1)
        
        if all_robots_finished_patrol:
            print(f"\n--- All robots completed their patrol cycles by step {step + 1}. ---")
            # Ensure final data is captured for any robot that just finished
            for robot in robots:
                 if robot.patrol_cycle_completed and (not robot.path_history or not np.array_equal(robot.path_history[-1][0], robot.position)):
                    # This case might be redundant if robot.move() handles it perfectly on completion
                    robot.path_history.append((robot.position.copy(), robot.sensed_signal_strength))
                    if robot.max_path_history > 0 and len(robot.path_history) > robot.max_path_history:
                        robot.path_history.pop(0)
            break 

        time.sleep(0.05) 
    
    if not all_robots_finished_patrol:
        print(f"\n--- Simulation finished after {SIMULATION_STEPS} steps. Not all robots may have completed their patrol. ---")
        # Capture final state for any robot that didn't finish but loop ended
        for robot in robots:
            if not robot.patrol_cycle_completed and (not robot.path_history or not np.array_equal(robot.path_history[-1][0], robot.position)):
                robot.path_history.append((robot.position.copy(), robot.sensed_signal_strength))
                if robot.max_path_history > 0 and len(robot.path_history) > robot.max_path_history:
                    robot.path_history.pop(0)


    for robot in robots:
        simulation_data.append({
            "robot_id": robot.robot_id,
            "path_history": list(robot.path_history) 
        })

    plt.ioff()
    plt.show()
    return simulation_data


def plot_signal_heatmap(simulation_data, map_vertices=None, signal_source_pos=None, min_signal=0, max_signal=1500):
    """
    Plots a heatmap of signal strengths from robot path data.

    Args:
        simulation_data (list): A list of dictionaries, where each dict has
                                'robot_id' and 'path_history'. Path_history is
                                a list of tuples: (np.array([x,y]), signal_strength).
        map_vertices (np.array, optional): Vertices of the map boundary to plot.
        signal_source_pos (np.array, optional): Position of the signal source to plot.
        min_signal (float): Minimum signal value for colormap normalization.
        max_signal (float): Maximum signal value for colormap normalization.
    """
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

    if not all_x_coords: # Check if any data points were actually extracted
        print("No path history points found in the simulation data.")
        return

    plt.figure(figsize=(10, 8))

    # Plot map boundary if provided
    if map_vertices is not None and len(map_vertices) > 0:
        map_poly = np.array(map_vertices)
        plt.plot(map_poly[:, 0], map_poly[:, 1], 'k-', label="Map Boundary", alpha=0.7)

    # Plot signal source if provided
    if signal_source_pos is not None:
        plt.plot(signal_source_pos[0], signal_source_pos[1], 'yx', markersize=12, markeredgewidth=2, label="Signal Source")

    # Create the scatter plot (heatmap effect)
    # Points are colored by their signal strength.
    # Using a colormap like 'viridis', 'plasma', 'inferno', or 'magma' often works well.
    cmap = plt.cm.viridis # Or plt.cm.get_cmap('viridis') in older matplotlib
    norm = Normalize(vmin=min_signal, vmax=max_signal) # Normalize signal strengths for color mapping

    scatter = plt.scatter(
        all_x_coords,
        all_y_coords,
        c=all_signal_strengths,
        cmap=cmap,
        norm=norm,
        s=15,  # Size of the points
        alpha=0.7, # Transparency of points
        edgecolors='none' # Can remove edgecolors for a smoother look
    )

    # Add a colorbar
    cbar = plt.colorbar(scatter, label="Sensed Signal Strength")
    
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Signal Strength Heatmap from Robot Paths")
    plt.axis('equal')  # Ensure aspect ratio is equal
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add legend for map boundary and signal source if they were plotted
    if (map_vertices is not None and len(map_vertices) > 0) or (signal_source_pos is not None):
        plt.legend(loc='upper right')

    plt.show()



if __name__ == "__main__":
    robot_data = run_simulation()

    plot_signal_heatmap(
        robot_data,
        map_vertices=np.array(cell.exterior.coords),
        signal_source_pos=SIGNAL_SOURCE_POS,
        min_signal=MIN_SIGNAL_FOR_COLORMAP,
        max_signal=MAX_SIGNAL_FOR_COLORMAP
    )