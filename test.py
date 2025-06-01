import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
# Load the simulation data
    try:
        df = pd.read_csv('simulation_data.csv')
    except FileNotFoundError:
        print("Error: 'simulation_data.csv' not found. Make sure the file is in the correct directory.")
        exit()

    # Constants and vertices as provided by the user
    SCALE = 2
    SHIFT = np.array([-5, -4])

    MAP_VERTICES = (
        np.array([
            (-2, 0), (11, 0), (11, -3), (14, -4), (17, -4),
            (20, -2.5), (20, 8), (17, 8), (17, 13), (13, 13),
            (13, 9), (4, 9), (4, 6), (-2, 6), (-2, 0)
        ]) * SCALE + SHIFT
    )

    PATROL_VERTICES = {
        0: np.array([
            (7.557, -0.031), (3.484, 3.194), (8.314, 5.109)
        ]),
        1: np.array([
            (23.186, -4.727), (25.840, -0.260), (28.382, -4.792)
        ]),
        2: np.array([
            (19.064, 1.362), (16.141, 5.658), (21.323, 6.041)
        ]),
    }

    def get_max_strength_on_edge(robot_id, v1, v2, df, tolerance=0.1):
        """
        Finds the point of maximum strength along a given edge for a specific robot.

        Args:
            robot_id (int): The ID of the robot.
            v1 (np.array): The start vertex of the edge.
            v2 (np.array): The end vertex of the edge.
            df (pd.DataFrame): The dataframe with simulation data.
            tolerance (float): The tolerance to consider a point on the edge.

        Returns:
            pd.Series or None: The data for the point with maximum strength, or None if no points are found.
        """
        robot_df = df[df['robot_id'] == robot_id].copy()

        # Vector representation of the edge
        edge_vec = v2 - v1
        edge_len_sq = np.sum(edge_vec**2)

        if edge_len_sq == 0:
            return None

        # Project each point onto the line defined by the edge
        robot_df['projection'] = ((robot_df[['x', 'y']].values - v1) * edge_vec).sum(axis=1) / edge_len_sq

        # Filter for points that project onto the segment
        on_segment_df = robot_df[(robot_df['projection'] >= 0) & (robot_df['projection'] <= 1)].copy()

        if on_segment_df.empty:
            return None

        # Calculate the orthogonal distance from each point to the line
        on_segment_df['distance'] = np.linalg.norm(np.cross(v2 - v1, on_segment_df[['x', 'y']].values - v1)) / np.linalg.norm(v2 - v1)

        # Filter for points within the tolerance
        edge_points_df = on_segment_df[on_segment_df['distance'] <= tolerance]

        if edge_points_df.empty:
            return None

        # Find the point with the maximum strength
        max_strength_point = edge_points_df.loc[edge_points_df['strength'].idxmax()]

        return max_strength_point

    def plot_analysis(orthogonal_line_length=2.0):
        """
        Analyzes and plots the robot patrol routes, highlighting maximum strength points.
        """
        fig, ax = plt.subplots(figsize=(15, 10))

        # Plot map
        map_path = np.vstack([MAP_VERTICES, MAP_VERTICES[0]])
        ax.plot(map_path[:, 0], map_path[:, 1], 'k-', label='Map Border')

        # Colors for each robot
        colors = ['r', 'g', 'b']
        robot_labels = {0: 'Robot 0', 1: 'Robot 1', 2: 'Robot 2'}

        for robot_id, vertices in PATROL_VERTICES.items():
            # Plot patrol graph
            patrol_path = np.vstack([vertices, vertices[0]])
            ax.plot(patrol_path[:, 0], patrol_path[:, 1], '--o', color=colors[robot_id], label=f'{robot_labels[robot_id]} Patrol Route')

            # Analyze each edge
            num_vertices = len(vertices)
            for i in range(num_vertices):
                v1 = vertices[i]
                v2 = vertices[(i + 1) % num_vertices] # Loop back to the start

                max_strength_point = get_max_strength_on_edge(robot_id, v1, v2, df)

                if max_strength_point is not None:
                    pos = np.array([max_strength_point['x'], max_strength_point['y']])

                    # Check if the max strength point is a vertex
                    is_vertex = np.allclose(pos, v1) or np.allclose(pos, v2)

                    if not is_vertex:
                        # Plot the max strength point
                        ax.plot(pos[0], pos[1], 'x', color=colors[robot_id], markersize=10, label=f'Max Strength (Robot {robot_id})')

                        # Calculate and plot orthogonal line
                        edge_vec = v2 - v1
                        if edge_vec[0] == 0: # Vertical line
                            ortho_vec = np.array([1, 0])
                        elif edge_vec[1] == 0: # Horizontal line
                            ortho_vec = np.array([0, 1])
                        else:
                            ortho_vec = np.array([-edge_vec[1], edge_vec[0]])

                        ortho_vec_norm = ortho_vec / np.linalg.norm(ortho_vec)
                        p1 = pos + ortho_vec_norm * orthogonal_line_length / 2.0
                        p2 = pos - ortho_vec_norm * orthogonal_line_length / 2.0

                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-', color=colors[robot_id], linewidth=2.5, label=f'Orthogonal Line (Robot {robot_id})')

        # Scatter plot of all robot positions for context
        for robot_id in range(3):
            robot_df = df[df['robot_id'] == robot_id]
            ax.scatter(robot_df['x'], robot_df['y'], s=robot_df['strength']/5, alpha=0.1, color=colors[robot_id])


        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Robot Patrol Route Analysis")
        ax.grid(True)
        ax.set_aspect('equal', adjustable='box')
        plt.show()

    # Perform the analysis and generate the plot
    plot_analysis(orthogonal_line_length=3.0)