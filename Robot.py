import math
import numpy as np

# Constants that the Robot class will use.
# These could also be passed into the constructor if more dynamic configuration is needed.
ROBOT_SPEED = 0.1
EPSILON = 1e-6
WAYPOINT_REACH_THRESHOLD = 0.15
ROBOT_ANGULAR_SPEED_RAD_PER_STEP = (
    math.pi / 18
)  # Radians per step (e.g., 10 degrees per step)
ORIENTATION_ALIGNMENT_THRESHOLD_RAD = (
    math.pi / 90
)  # Tolerance for alignment (e.g., 2 degrees)


class Robot:
    def __init__(
        self,
        robot_id,
        waypoints,
        source,
        initial_pos=None,
        speed=ROBOT_SPEED,
        max_history=0,
        sense_type="Omni",
    ):
        """
        Initializes a Robot instance.

        Args:
            robot_id (int): Unique identifier for the robot.
            waypoints (np.array): A 2D numpy array of [x, y] coordinates for the robot's patrol route.
            source (dict): Dictionary containing signal source info, e.g., {"pos": np.array, "params": dict}.
            initial_pos (np.array, optional): Initial [x, y] position. Defaults to the first waypoint.
            speed (float, optional): Movement speed of the robot. Defaults to ROBOT_SPEED.
            max_history (int, optional): Maximum length of path history. 0 for unlimited.
            sense_type (str, optional): Type of sensor, "Omni" or "Dir". Defaults to "Omni".
        """
        self.robot_id = robot_id
        self.waypoints = (
            np.array(waypoints) if waypoints is not None else np.array([])
        )  # Ensure waypoints is a numpy array
        self.current_waypoint_index = 0

        if initial_pos is None and self.waypoints.size > 0:
            self.position = np.array(self.waypoints[0], dtype=float)
        elif initial_pos is not None:
            self.position = np.array(initial_pos, dtype=float)
        else:
            self.position = np.array([0.0, 0.0], dtype=float)
            print(
                f"Warning: Robot {robot_id} initialized at (0,0) due to no waypoints or initial_pos being None/empty."
            )

        self.speed = speed
        self.sensed_signal_strength = 0.0  # Initialized as float, will be updated by sense_signal
        self.orientation = 0.0  # Radians, 0 is along positive X-axis
        self.is_rotating = False
        self.departure_orientation = 0.0  # Target orientation after rotation

        self.initial_patrol_waypoint_pos = (
            self.waypoints[0].copy()
            if self.waypoints.size > 0
            else self.position.copy()
        )
        self.has_left_initial_waypoint = False
        self.patrol_cycle_completed = False

        self.max_path_history = max_history

        if self.waypoints.size > 0:
            first_target_wp = self.waypoints[self.current_waypoint_index]
            if np.linalg.norm(self.position - first_target_wp) < EPSILON:
                if len(self.waypoints) > 1:
                    next_target_wp = self.waypoints[1]
                    direction_vector_to_next = next_target_wp - self.position
                    if np.linalg.norm(direction_vector_to_next) > EPSILON:
                        self.orientation = math.atan2(
                            direction_vector_to_next[1], direction_vector_to_next[0]
                        )
            else:
                direction_vector = first_target_wp - self.position
                if np.linalg.norm(direction_vector) > EPSILON:
                    self.orientation = math.atan2(
                        direction_vector[1], direction_vector[0]
                    )
        self.departure_orientation = self.normalize_angle(self.orientation)

        self.sense_type = sense_type
        
        if len(self.waypoints) == 3:
            offset_angle = 30
        elif len(self.waypoints) == 4:
            offset_angle = 45
        elif len(self.waypoints) == 6:
            offset_angle = 60
        else:
            offset_angle = 0 # Default or for other cases
        
        self.offsets = np.array([np.radians(-offset_angle), np.radians(offset_angle)])
        
        self.sense_signal(source["pos"], source["params"])
        
        # Store a copy of the signal strength if it's a numpy array
        initial_signal_to_store = (
            self.sensed_signal_strength.copy()
            if isinstance(self.sensed_signal_strength, np.ndarray)
            else self.sensed_signal_strength
        )
        self.path_history = [(self.position.copy(), initial_signal_to_store)]

    def normalize_angle(self, angle):
        """Normalize an angle to the range [-pi, pi]."""
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def _signals_equal(self, s1, s2):
        """Checks if two signal strength values are equal, handling scalars and numpy arrays."""
        if isinstance(s1, np.ndarray) and isinstance(s2, np.ndarray):
            return np.array_equal(s1, s2)
        # If one is an array and the other is not, they are not equal
        if isinstance(s1, np.ndarray) or isinstance(s2, np.ndarray):
            return False
        # Both are scalars (or types that support direct equality)
        return s1 == s2

    def move(self):
        """
        Moves the robot. Handles movement towards waypoints, rotation at waypoints,
        and patrol completion logic. Appends current state to path_history.
        """
        if self.patrol_cycle_completed or not self.waypoints.size:
            if (
                not self.path_history
                or not np.array_equal(self.path_history[-1][0], self.position)
                or not self._signals_equal(self.path_history[-1][1], self.sensed_signal_strength) # Use helper
            ):
                self._append_to_path_history()
            return

        current_wp_is_initial = np.array_equal(
            self.waypoints[self.current_waypoint_index],
            self.initial_patrol_waypoint_pos,
        )

        if self.is_rotating:
            angle_difference = self.normalize_angle(
                self.departure_orientation - self.orientation
            )
            if abs(angle_difference) < ORIENTATION_ALIGNMENT_THRESHOLD_RAD:
                self.orientation = self.normalize_angle(self.departure_orientation)
                self.is_rotating = False
                if current_wp_is_initial and not self.has_left_initial_waypoint:
                    self.has_left_initial_waypoint = True
                self.current_waypoint_index = (self.current_waypoint_index + 1) % len(
                    self.waypoints
                )
            else:
                turn_direction = np.sign(angle_difference)
                rotation_step = turn_direction * ROBOT_ANGULAR_SPEED_RAD_PER_STEP
                if abs(rotation_step) > abs(angle_difference):
                    self.orientation = self.normalize_angle(self.departure_orientation)
                else:
                    self.orientation = self.normalize_angle(
                        self.orientation + rotation_step
                    )
        else: 
            target_waypoint = self.waypoints[self.current_waypoint_index]
            direction_vector = target_waypoint - self.position
            distance_to_waypoint = np.linalg.norm(direction_vector)

            if distance_to_waypoint < WAYPOINT_REACH_THRESHOLD:
                self.position = target_waypoint.copy()
                if (
                    np.array_equal(target_waypoint, self.initial_patrol_waypoint_pos)
                    and self.has_left_initial_waypoint
                ):
                    self.patrol_cycle_completed = True
                    self._append_to_path_history() 
                    return 

                next_wp_idx = (self.current_waypoint_index + 1) % len(self.waypoints)
                next_target_wp = self.waypoints[next_wp_idx]

                if (
                    np.array_equal(target_waypoint, next_target_wp)
                    or len(self.waypoints) < 2
                ):
                    self.departure_orientation = self.normalize_angle(self.orientation)
                    self.is_rotating = False 
                    if current_wp_is_initial and not self.has_left_initial_waypoint:
                        self.has_left_initial_waypoint = True
                    self.current_waypoint_index = next_wp_idx
                else: 
                    direction_to_next_target = next_target_wp - self.position
                    self.departure_orientation = self.normalize_angle(
                        math.atan2(
                            direction_to_next_target[1], direction_to_next_target[0]
                        )
                    )
                    current_orientation_normalized = self.normalize_angle(self.orientation)
                    angle_diff_to_departure = self.normalize_angle(
                        self.departure_orientation - current_orientation_normalized
                    )
                    if (
                        abs(angle_diff_to_departure)
                        > ORIENTATION_ALIGNMENT_THRESHOLD_RAD
                    ):
                        self.is_rotating = True
                    else: 
                        self.orientation = self.departure_orientation
                        self.is_rotating = False
                        if current_wp_is_initial and not self.has_left_initial_waypoint:
                            self.has_left_initial_waypoint = True
                        self.current_waypoint_index = next_wp_idx
            elif (
                distance_to_waypoint > EPSILON
            ): 
                self.orientation = self.normalize_angle(
                    math.atan2(direction_vector[1], direction_vector[0])
                )
                move_vector = (direction_vector / distance_to_waypoint) * self.speed
                self.position += move_vector
                if not self.has_left_initial_waypoint and not current_wp_is_initial:
                    self.has_left_initial_waypoint = True
        self._append_to_path_history()

    def _append_to_path_history(self):
        """Helper method to append current state to path_history and manage its length."""
        # Store a copy of the signal strength if it's a numpy array
        current_signal_to_store = (
            self.sensed_signal_strength.copy()
            if isinstance(self.sensed_signal_strength, np.ndarray)
            else self.sensed_signal_strength
        )

        if (
            not self.path_history
            or not np.array_equal(self.path_history[-1][0], self.position)
            # Use the helper method for comparing signal strengths
            or not self._signals_equal(self.path_history[-1][1], self.sensed_signal_strength)
        ):
            self.path_history.append(
                (self.position.copy(), current_signal_to_store)
            )
            if (
                self.max_path_history > 0
                and len(self.path_history) > self.max_path_history
            ):
                self.path_history.pop(0)

    def sense_signal(self, signal_source_pos, emitter_params):

        def convert_to_dB(value):
            # Handles both scalar and numpy array inputs for value
            return 10 * np.log10(np.asarray(value) + EPSILON) # Added EPSILON to avoid log(0)

        def angle_to_point(agent_pos, agent_theta, target_pos):
            dx = target_pos[0] - agent_pos[0]
            dy = target_pos[1] - agent_pos[1]
            target_angle = math.atan2(dy, dx)
            angle_diff = (target_angle - agent_theta + math.pi) % (
                2 * math.pi
            ) - math.pi
            return angle_diff

        efficiency = emitter_params["efficiency"]
        pt = emitter_params["pt"]
        gain_t = emitter_params["gain"] # Assuming this is a scalar gain for the transmitter
        wavelength = emitter_params["wavelength"]

        distance = np.linalg.norm(self.position - signal_source_pos)
        if distance < EPSILON: # Avoid division by zero if at the source
            distance = EPSILON

        if self.sense_type == "Omni":
            # For Omni, gain_array is a scalar, los is a scalar
            gain_array_val = 1.0 
            los_val = angle_to_point(self.position, self.orientation, signal_source_pos)
            # Ensure radiation calculation results in scalar for Omni
            radiation = 1.0 # Omni is always 1.0, no sinc pattern
        elif self.sense_type == "Dir":
            # For Dir, gain_array is an array, los is a list (becomes array in calculation)
            gain_array_val = np.array([4.0, 4.0]) # Example directional gain
            los_val = np.array([angle_to_point(self.position, self.orientation + offset, signal_source_pos) for offset in self.offsets])
            # Radiation calculation for directional antenna
            # Ensure sinc argument is not zero, add EPSILON if necessary, though pi in denominator helps
            sinc_arg = (gain_array_val * los_val) / (np.pi + EPSILON)
            radiation = gain_array_val * (np.sinc(sinc_arg) ** 2)
        else:
            raise ValueError(f"Unknown sense_type: {self.sense_type}")
            
        noise = 0 # np.random.normal(0, 0.1) # Example noise, can be array if radiation is array

        # Free Space Path Loss (FSPL) in dB
        fspl_db = convert_to_dB((wavelength / (4 * np.pi * distance))**2) # FSPL formula P_r/P_t = G_t*G_r*(lambda/(4*pi*d))^2

        # Received power in dBm or dB
        # Pr = Pt_dB + Gt_dB + Gr_dB - FSPL_dB (incorrect formula structure previously)
        # Pr_watts = Pt_watts * Gt * Gr * (lambda / (4*pi*d))^2
        # Pr_dB = Pt_dB + Gt_dB + Gr_dB + 20*log10(lambda / (4*pi*d))
        # Or Pr_dB = Pt_dB + Gt_dB + Gr_dB + 20*log10(lambda) - 20*log10(4*pi*d)

        # Assuming emitter_params["pt"] is in dB (e.g., dBm)
        # Assuming emitter_params["gain"] is transmitter gain in dBi
        # 'radiation' here acts as the relative receiver gain pattern (Gr)
        
        pr = (
            pt  # Transmitted power (e.g., in dBm)
            + gain_t # Transmitter gain (e.g., in dBi)
            + convert_to_dB(radiation) # Receiver gain pattern (now in dB)
            + fspl_db # Path loss (already in dB and typically negative)
            + noise # Noise in dB
        )
        # No, efficiency should be applied to linear values or handled in dB appropriately.
        # If efficiency is a factor (0-1), apply it before dB conversion or subtract -10*log10(efficiency_factor)
        # For now, assuming pt, gain_t are in dB, radiation becomes dB.
        # The previous formula for pr was adding dB values somewhat inconsistently.
        # A more standard link budget: Pr_dB = Pt_dB + Gt_dB + Gr_dB - L_path_dB - L_other_dB
        # Where Gr_dB here is convert_to_dB(radiation)
        # And L_path_dB is -fspl_db if fspl_db is defined as 20log10(lambda/(4pid))
        # Let's re-evaluate based on common FSPL application.
        # FSPL_dB = 20*log10(d) + 20*log10(f) + 20*log10(4pi/c) - Gt - Gr  (if f is freq)
        # Or simpler: L_fspl = -20*log10(lambda / (4*pi*d)) = 20*log10(4*pi*d/lambda)

        # Corrected Received Power Calculation (simplified, assuming pt and gain_t are in dB)
        # Let Pr_linear = Pt_linear * Gt_linear * Gr_linear * (lambda / (4 * pi * d))^2 * efficiency
        # So Pr_dB = Pt_dB + Gt_dB + Gr_dB + 20*log10(lambda / (4*pi*d)) + 10*log10(efficiency) + Noise_dB

        pt_linear = 10**(pt/10) # Convert Pt from dB to linear
        gain_t_linear = 10**(gain_t/10) # Convert Gt from dB to linear
        
        # radiation is already gain_r_linear_pattern
        # distance factor squared for power
        path_loss_linear_factor = (wavelength / (4 * np.pi * distance + EPSILON))**2

        pr_linear = efficiency * pt_linear * gain_t_linear * radiation * path_loss_linear_factor
        
        # Convert final received power to dB, then add noise in dB
        self.sensed_signal_strength = convert_to_dB(pr_linear) + noise
        
        # If you want to keep the original structure with adding dB values:
        # Be careful with convert_to_dB(radiation) if radiation can be 0. Added EPSILON in convert_to_dB.
        # Original-like structure attempt (ensure all terms are compatible):
        # log_lambda_factor = convert_to_dB(wavelength / (4 * np.pi * distance + EPSILON)) # This is 10log10, for power should be 20log10 if lambda factor is not squared.
        # path_gain_db = 2 * 10 * np.log10(wavelength / (4 * np.pi * distance + EPSILON)) # 20log10(lambda / (4pid))
        
        # pr_final_attempt_db = (
        #     pt # Pt in dB
        #     + gain_t # Gt in dB
        #     + convert_to_dB(radiation) # Gr pattern in dB
        #     + path_gain_db # Path gain in dB (includes lambda, d)
        #     + 10 * np.log10(efficiency + EPSILON) # Efficiency in dB
        #     + noise # Noise in dB
        # )
        # self.sensed_signal_strength = pr_final_attempt_db

        return self.sensed_signal_strength
    
    def update_patrol(self, new_waypoints):
            """
            Updates the robot's patrol route and resets its state to begin the new route.
            The robot will first travel from its current position to the start of the new route.
            """
            self.waypoints = np.array(new_waypoints) if new_waypoints is not None else np.array([])
            self.current_waypoint_index = 0
            self.patrol_cycle_completed = False
            self.is_rotating = False

            if self.waypoints.size > 0:
                self.initial_patrol_waypoint_pos = self.waypoints[0].copy()
                self.has_left_initial_waypoint = False
            else:
                self.initial_patrol_waypoint_pos = self.position.copy()
                self.patrol_cycle_completed = True
                self.has_left_initial_waypoint = True
            
    def __str__(self):
        """String representation of the Robot's current state."""
        state = (
            "Completed"
            if self.patrol_cycle_completed
            else ("Rotating" if self.is_rotating else "Moving")
        )
        orientation_deg = (
            math.degrees(self.orientation) % 360
        )

        # Handle scalar or numpy array for sensed_signal_strength
        if isinstance(self.sensed_signal_strength, np.ndarray):
            # Format numpy array to string with 2 decimal places for each element
            signal_str = np.array2string(self.sensed_signal_strength, formatter={'float_kind':lambda x: "%.2f" % x})
        elif isinstance(self.sensed_signal_strength, (float, np.floating)):
             # Format scalar float to string with 2 decimal places
            signal_str = f"{self.sensed_signal_strength:.2f}"
        else:
            signal_str = str(self.sensed_signal_strength) # Fallback for other types

        return (
            f"Robot {self.robot_id}: Pos=({self.position[0]:.2f}, {self.position[1]:.2f}), State={state}, "
            f"Ori={orientation_deg:.0f}°, TargetWP_Idx={self.current_waypoint_index}, "
            f"Signal={signal_str}"
        )