import math
import numpy as np

# Constants that the Robot class will use.
# These could also be passed into the constructor if more dynamic configuration is needed.
ROBOT_SPEED = 0.1
EPSILON = 1e-6
WAYPOINT_REACH_THRESHOLD = 0.15
ROBOT_ANGULAR_SPEED_RAD_PER_STEP = math.pi / 18 # Radians per step (e.g., 10 degrees per step)
ORIENTATION_ALIGNMENT_THRESHOLD_RAD = math.pi / 90 # Tolerance for alignment (e.g., 2 degrees)

class Robot:
    def __init__(self, robot_id, waypoints, source, initial_pos=None, speed=ROBOT_SPEED, max_history=0):
        """
        Initializes a Robot instance.

        Args:
            robot_id (int): Unique identifier for the robot.
            waypoints (np.array): A 2D numpy array of [x, y] coordinates for the robot's patrol route.
            source (dict): Dictionary containing signal source info, e.g., {"pos": np.array, "strength": float}.
            initial_pos (np.array, optional): Initial [x, y] position. Defaults to the first waypoint.
            speed (float, optional): Movement speed of the robot. Defaults to ROBOT_SPEED.
            max_history (int, optional): Maximum length of path history. 0 for unlimited.
        """
        self.robot_id = robot_id
        self.waypoints = np.array(waypoints) if waypoints is not None else np.array([]) # Ensure waypoints is a numpy array
        self.current_waypoint_index = 0
        
        if initial_pos is None and self.waypoints.size > 0:
            self.position = np.array(self.waypoints[0], dtype=float)
        elif initial_pos is not None:
            self.position = np.array(initial_pos, dtype=float)
        else:
            # This case should ideally be handled by ensuring waypoints are always provided
            # or by setting a more sensible default if waypoints can be empty.
            self.position = np.array([0.0, 0.0], dtype=float)
            print(f"Warning: Robot {robot_id} initialized at (0,0) due to no waypoints or initial_pos being None/empty.")

        self.speed = speed
        self.sensed_signal_strength = 0.0
        self.orientation = 0.0  # Radians, 0 is along positive X-axis
        self.is_rotating = False
        self.departure_orientation = 0.0 # Target orientation after rotation

        # Patrol completion tracking
        self.initial_patrol_waypoint_pos = self.waypoints[0].copy() if self.waypoints.size > 0 else self.position.copy()
        self.has_left_initial_waypoint = False # True once it moves from/rotates at the first WP
        self.patrol_cycle_completed = False    # True once it returns to the first WP after starting
        
        self.max_path_history = max_history

        # Initialize orientation based on the first or second waypoint
        if self.waypoints.size > 0:
            first_target_wp = self.waypoints[self.current_waypoint_index]
            # Check if starting at the first waypoint
            if np.linalg.norm(self.position - first_target_wp) < EPSILON:
                if len(self.waypoints) > 1: # If at WP0 and there's a WP1
                    next_target_wp = self.waypoints[1]
                    direction_vector_to_next = next_target_wp - self.position
                    if np.linalg.norm(direction_vector_to_next) > EPSILON: # Check if next WP is different
                        self.orientation = math.atan2(direction_vector_to_next[1], direction_vector_to_next[0])
                # If only one waypoint and starting at it, orientation remains 0 (or could be arbitrary)
            else: # Starting away from the first waypoint, orient towards it
                direction_vector = first_target_wp - self.position
                if np.linalg.norm(direction_vector) > EPSILON:
                     self.orientation = math.atan2(direction_vector[1], direction_vector[0])
        self.departure_orientation = self.normalize_angle(self.orientation) # Initial departure is current orientation
        
        # Initial signal sense
        self.sense_signal(source["pos"], source["strength"])
        self.path_history = [(self.position.copy(), self.sensed_signal_strength)]


    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def move(self):
        """
        Moves the robot. Handles movement towards waypoints, rotation at waypoints,
        and patrol completion logic. Appends current state to path_history.
        """
        # If patrol is completed or no waypoints, do nothing but ensure path history is up-to-date for the final state
        if self.patrol_cycle_completed or not self.waypoints.size:
            # Ensure the very last state is recorded if it wasn't during the completion step
            if not self.path_history or not np.array_equal(self.path_history[-1][0], self.position) or self.path_history[-1][1] != self.sensed_signal_strength:
                self._append_to_path_history()
            return

        current_wp_is_initial = np.array_equal(self.waypoints[self.current_waypoint_index], self.initial_patrol_waypoint_pos)

        if self.is_rotating:
            angle_difference = self.normalize_angle(self.departure_orientation - self.orientation)
            if abs(angle_difference) < ORIENTATION_ALIGNMENT_THRESHOLD_RAD:
                self.orientation = self.normalize_angle(self.departure_orientation) # Snap to final orientation
                self.is_rotating = False
                
                # Mark as having left initial waypoint after first rotation at start
                if current_wp_is_initial and not self.has_left_initial_waypoint:
                    self.has_left_initial_waypoint = True

                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(self.waypoints)
            else:
                turn_direction = np.sign(angle_difference)
                rotation_step = turn_direction * ROBOT_ANGULAR_SPEED_RAD_PER_STEP
                # Prevent overshooting the target orientation
                if abs(rotation_step) > abs(angle_difference):
                    self.orientation = self.normalize_angle(self.departure_orientation)
                else:
                    self.orientation = self.normalize_angle(self.orientation + rotation_step)
        else: # Not rotating: either moving or deciding to rotate/complete
            target_waypoint = self.waypoints[self.current_waypoint_index]
            direction_vector = target_waypoint - self.position
            distance_to_waypoint = np.linalg.norm(direction_vector)

            if distance_to_waypoint < WAYPOINT_REACH_THRESHOLD: # Reached a waypoint
                # Waypoint snapping is intentionally omitted as per previous request.
                # Position remains as it arrived within the threshold.

                # Check for patrol completion: at initial waypoint AND has already left it once.
                if np.array_equal(target_waypoint, self.initial_patrol_waypoint_pos) and self.has_left_initial_waypoint:
                    self.patrol_cycle_completed = True
                    self._append_to_path_history() # Record final state upon completion
                    return # Stop further processing for this step

                # Determine next waypoint and if rotation is needed
                next_wp_idx = (self.current_waypoint_index + 1) % len(self.waypoints)
                next_target_wp = self.waypoints[next_wp_idx]
                
                # If current waypoint is the same as next (e.g., single waypoint patrol or last returning to same point)
                # or if there's only one waypoint in the list.
                if np.array_equal(target_waypoint, next_target_wp) or len(self.waypoints) < 2:
                    self.departure_orientation = self.normalize_angle(self.orientation) # Maintain current orientation
                    self.is_rotating = False # No rotation needed
                    if current_wp_is_initial and not self.has_left_initial_waypoint:
                         self.has_left_initial_waypoint = True # Considered "left" if moving on from initial WP
                    self.current_waypoint_index = next_wp_idx
                else: # Calculate departure orientation for the next distinct waypoint
                    # Use current robot position for calculating direction to next target
                    direction_to_next_target = next_target_wp - self.position 
                    self.departure_orientation = self.normalize_angle(math.atan2(direction_to_next_target[1], direction_to_next_target[0]))
                    
                    current_orientation_normalized = self.normalize_angle(self.orientation)
                    angle_diff_to_departure = self.normalize_angle(self.departure_orientation - current_orientation_normalized)

                    if abs(angle_diff_to_departure) > ORIENTATION_ALIGNMENT_THRESHOLD_RAD:
                        self.is_rotating = True # Rotation is needed
                    else: # Already aligned
                        self.orientation = self.departure_orientation # Snap to exact if very close
                        self.is_rotating = False
                        if current_wp_is_initial and not self.has_left_initial_waypoint:
                            self.has_left_initial_waypoint = True
                        self.current_waypoint_index = next_wp_idx
            
            elif distance_to_waypoint > EPSILON: # Moving towards the current target_waypoint
                # Update orientation to face the target waypoint while moving
                self.orientation = self.normalize_angle(math.atan2(direction_vector[1], direction_vector[0]))
                
                move_vector = (direction_vector / distance_to_waypoint) * self.speed
                
                # Move the robot. Without snapping, it might slightly overshoot or undershoot the threshold.
                # The WAYPOINT_REACH_THRESHOLD handles arrival.
                self.position += move_vector

                # Logic for has_left_initial_waypoint when starting away from WP0:
                # It's considered "left" once it arrives at WP0 and then either rotates or moves to WP1.
                # If it's moving towards a non-initial waypoint, it implies it must have "left" the initial sequence.
                if not self.has_left_initial_waypoint and not current_wp_is_initial:
                    self.has_left_initial_waypoint = True

        # Append current state to path history after any action (move/rotate)
        self._append_to_path_history()

    def _append_to_path_history(self):
        """Helper method to append current state to path_history and manage its length."""
        # Avoid appending duplicate states if no actual change occurred
        if not self.path_history or \
           not np.array_equal(self.path_history[-1][0], self.position) or \
           self.path_history[-1][1] != self.sensed_signal_strength:
            
            self.path_history.append((self.position.copy(), self.sensed_signal_strength))
            if self.max_path_history > 0 and len(self.path_history) > self.max_path_history:
                self.path_history.pop(0) # Remove the oldest element

    def sense_signal(self, signal_source_pos, signal_power):
        """
        Senses the signal strength from a given source.
        Updates self.sensed_signal_strength.

        Args:
            signal_source_pos (np.array): [x, y] position of the signal source.
            signal_power (float): Power of the signal source.

        Returns:
            float: The sensed signal strength.
        """
        distance_to_source = np.linalg.norm(self.position - signal_source_pos)
        self.sensed_signal_strength = signal_power / (distance_to_source**2 + EPSILON)
        return self.sensed_signal_strength

    def __str__(self):
        """String representation of the Robot's current state."""
        state = "Completed" if self.patrol_cycle_completed else ("Rotating" if self.is_rotating else "Moving")
        orientation_deg = math.degrees(self.orientation) % 360 # Ensure positive degrees
        return (f"Robot {self.robot_id}: Pos=({self.position[0]:.2f}, {self.position[1]:.2f}), State={state}, "
                f"Ori={orientation_deg:.0f}°, TargetWP_Idx={self.current_waypoint_index}, "
                f"Signal={self.sensed_signal_strength:.2f}")
