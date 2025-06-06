import os
import csv
import numpy as np
from Simulator import Simulator
import matplotlib.pyplot as plt
from Patrol import OmniPatrol, DirPatrol
from shapely.geometry import Polygon, Point
from scipy.optimize import differential_evolution
from optimize import generate_bounds, generalized_objective
from Voronoi import create_voronoi_partitions_on_map, plot_results

SHAPE_CONFIG = {"Triangle": 3, "Square": 0, "Hexagon": 0}
PATROL_TYPE = OmniPatrol
NUM_TRIALS = 5
S_R = 8.0
DE_WEIGHTS = [2.0, 1.0, 0.5, 1.0, 10.0]

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

C = 3e8
EMITTER_PARAMS = {"efficiency": 1.0, "pt": 30, "gain": 1.0, "wavelength": C / 2.4e9}


def random_point_in_polygon(polygon_vertices):
    poly = Polygon(polygon_vertices)
    minx, miny, maxx, maxy = poly.bounds
    while True:
        p = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
        if poly.contains(p):
            return np.array([p.x, p.y])


def voronoi_cells(map_polygon, x, y, plot=True):
    minx, miny, maxx, maxy = map_polygon.bounds
    x_coords = np.linspace(minx, maxx, x)
    y_coords = np.linspace(miny, maxy, y)
    xbound = (minx, maxx)
    ybound = (miny, maxy)
    grid_points_input = []
    for x_coord in x_coords:
        for y_coord in y_coords:
            grid_points_input.append((x_coord, y_coord))
    clipped_cells, used_sites = create_voronoi_partitions_on_map(
        map_polygon, grid_points_input
    )
    if plot:
        plot_results(map_polygon, clipped_cells, used_sites)
    return clipped_cells, xbound, ybound


def area_coverage_optimiser(cell, xbound, ybound, PatrolType):
    if PatrolType == OmniPatrol:
        patrol_type = "Omni"
    elif PatrolType == DirPatrol:
        patrol_type = "Dir"
    else:
        raise ValueError("PatrolType must be OmniPatrol or DirPatrol.")
    patrols = []
    bounds = generate_bounds(SHAPE_CONFIG, S_R, xbound, ybound, patrol=patrol_type)
    print("Optimizer Started")
    result = differential_evolution(
        generalized_objective,
        bounds,
        args=(S_R, DE_WEIGHTS, SHAPE_CONFIG, cell, PatrolType),
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


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists("data"):
        os.makedirs("data")
    
    summary_file_path = os.path.join("data", "simulation_summary.txt")

    # Write the header with map vertices to the summary file (overwriting any previous file)
    with open(summary_file_path, "w") as f:
        f.write("--- Simulation Environment ---\n")
        map_vertices_str = np.array2string(MAP_VERTICES, separator=', ').replace('\n', '')
        f.write(f"Map Vertices: {map_vertices_str}\n")
        f.write("="*40 + "\n\n")


    for trial in range(NUM_TRIALS):
        print(f"\n--- Starting Trial {trial} ---")
        SIGNAL_SOURCE_POS = random_point_in_polygon(MAP_VERTICES)
        PLOTTING = False  # Disable plotting for batch trials
        map_polygon = Polygon(MAP_VERTICES)
        cells, xbound, ybound = voronoi_cells(map_polygon, 4, 4, plot=PLOTTING)

        if PATROL_TYPE == OmniPatrol:
            sense_type = "Omni"
        elif PATROL_TYPE == DirPatrol:
            sense_type = "Dir"

        params = {
            "sim_steps": 500,
            "map_vertices": MAP_VERTICES,
            "num_robots": sum(SHAPE_CONFIG.values()),
            "source": {
                "pos": SIGNAL_SOURCE_POS,
                "params": EMITTER_PARAMS,
            },
            "sense_type": sense_type,
        }
        
        simulator = Simulator(params, plotting=PLOTTING, sleep_time=0.0)
        
        trial_sim_data = []
        total_steps_for_trial = 0
        all_patrol_waypoints = {} # Store waypoints for the summary

        for i, cell in enumerate(cells):
            print(f"\nRunning simulation for Cell {i} in Trial {trial}...")
            opt_params, patrols = area_coverage_optimiser(
                cell,
                xbound,
                ybound,
                PatrolType=PATROL_TYPE,
            )
            
            robot_waypoints = [np.array(p.patrol.exterior.coords) for p in patrols]
            all_patrol_waypoints[i] = robot_waypoints # Save for logging

            if PLOTTING:
                plt.ion()

            sim_data_for_cell, step_count = simulator.run(
                {"cell_id": i, "cell_polygon": cell}, robot_waypoints
            )
            trial_sim_data.extend(sim_data_for_cell)
            total_steps_for_trial += step_count

        # --- Data and Summary Writing for the Trial ---
        
        # 1. Write aggregated data to a CSV file
        csv_file_name = f"{sense_type}_trial_{trial}.csv"
        csv_file_path = os.path.join("data", csv_file_name)
        with open(csv_file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["cell_id", "robot_id", "x", "y", "orientation", "signal_strength"])
            for entry in trial_sim_data:
                cell_id = entry["cell_id"]
                robot_id = entry["robot_id"]
                for pos, ori, sig in entry["path_history"]:
                    if isinstance(sig, (list, np.ndarray)):
                        strength = ";".join(map(str, sig))
                    else:
                        strength = str(sig)
                    writer.writerow([cell_id, robot_id, pos[0], pos[1], ori, strength])
        
        print(f"\nTrial {trial} data saved to {csv_file_path}")

        # 2. Prepare the summary string for the text file and append it
        with open(summary_file_path, "a") as f:
            f.write(f"--- Trial Number: {trial} ---\n")
            f.write(f"Total Steps (sum over cells): {total_steps_for_trial}\n")
            f.write(f"Associated CSV File: {csv_file_name}\n")
            f.write(f"Source Position: {np.array2string(SIGNAL_SOURCE_POS, precision=2)}\n")
            f.write(f"Emitter Parameters: {EMITTER_PARAMS}\n")
            f.write(f"S_R: {S_R}\n")
            f.write(f"Number of Cells: {len(cells)}\n")
            f.write("Patrol Vertices per Cell:\n")
            for cell_id, waypoints_list in all_patrol_waypoints.items():
                f.write(f"  Cell {cell_id}:\n")
                for robot_idx, wps in enumerate(waypoints_list):
                    wps_str = np.array2string(wps, separator=', ', prefix=f'    Robot {robot_idx}: ')
                    f.write(f"    Robot {robot_idx}: {wps_str}\n")

            f.write("----------------------------------------\n\n")

        print(f"--- Finished Trial {trial} ---")

    print(f"\nAll trials complete. Summary log written to {summary_file_path}")