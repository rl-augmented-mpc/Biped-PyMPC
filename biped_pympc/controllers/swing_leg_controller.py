import torch
from time import time
from typing import Literal

from biped_pympc.core.robot.robot_factory import RobotFactory
from biped_pympc.core.data.robot_data import StateEStimatorData, DesiredStateData, LegControllerData
from biped_pympc.controllers.swing_leg_trajectory import SwingLegTrajectory

@torch.jit.script
def compute_raibert_heuristic_placement(
    root_position: torch.Tensor, 
    rotation_body: torch.Tensor,
    root_velocity_w: torch.Tensor,
    root_velocity_w_desired: torch.Tensor,
    swing_time: torch.Tensor,
    swing_time_remaining: torch.Tensor,
    max_x: float,
    max_y: float,
    kx: float,
    ky: float,
    hip_position: torch.Tensor,
)->torch.Tensor:
    """_summary_

    Args:
        root_position (torch.Tensor): [batch_size, 3]
        hip_yaw_position (torch.Tensor): [batch_size, num_legs, 3]
        root_velocity_w (torch.Tensor): [batch_size, 3]
        root_velocity_w_desired (torch.Tensor): [batch_size
        rotation_body (torch.Tensor): [batch_size, 3, 3]
        swing_time_remaining (torch.Tensor): [batch_size, num_legs]
        max_x (float): [description]
        max_y (float): [description]
        kx (float): [description]
        ky (float): [description]
    """
    
    device = root_position.device
    batch_size = root_position.shape[0]
    num_legs = hip_position.shape[1]
    
    # compute the foot position in world frame
    foot_placement = torch.zeros((batch_size, num_legs, 3), device=device)
    foot_placement[:, 0, :] = root_position + torch.bmm(rotation_body, hip_position[:, 0, :].unsqueeze(-1)).squeeze(-1) + \
        0.5 * root_velocity_w * swing_time_remaining[:, 0].unsqueeze(-1)
    foot_placement[:, 1, :] = root_position + torch.bmm(rotation_body, hip_position[:, 1, :].unsqueeze(-1)).squeeze(-1) + \
        0.5* root_velocity_w * swing_time_remaining[:, 1].unsqueeze(-1)
    
    # feedback term
    foot_placement_fb = torch.zeros((batch_size, num_legs, 3), device=device)
    foot_placement_fb[:, 0, 0] = kx * (root_velocity_w[:, 0] - root_velocity_w_desired[:, 0])
    foot_placement_fb[:, 1, 0] = kx * (root_velocity_w[:, 0] - root_velocity_w_desired[:, 0])
    foot_placement_fb[:, 0, 1] = ky * (root_velocity_w[:, 1] - root_velocity_w_desired[:, 1])
    foot_placement_fb[:, 1, 1] = ky * (root_velocity_w[:, 1] - root_velocity_w_desired[:, 1])
    foot_placement_fb[:, :, 0] = torch.clamp(foot_placement_fb[:, :, 0], -max_x, max_x)
    foot_placement_fb[:, :, 1] = torch.clamp(foot_placement_fb[:, :, 1], -max_y, max_y)
    
    foot_placement_raibert = foot_placement + foot_placement_fb
    foot_placement_raibert[:, :, 2] = 0.0 # TODO: accept z position
    
    return foot_placement_raibert

class SwingLegController:
    def __init__(self, 
                 dt: float, 
                 batch_size: int, 
                 num_legs:int, 
                 device: torch.device, 
                 swing_duration:torch.Tensor, 
                 swing_height:float=0.1, 
                 reference_frame: Literal["world", "base"]="base",
                 robot: Literal["HECTOR", "T1"]="HECTOR"):
        """
        args:
            dt: time step
            batch_size: number of environments
            num_legs: number of legs
            device: device to use (CPU or GPU)
            swing_duration: duration of the swing phase for each leg in second [batch_size, num_legs]
            swing_height: height of the swing trajectory
            reference_frame: "world" or "base", the frame in which the foot placement is computed
            robot: robot model to use
        """
        self.dt = dt
        self.batch_size = batch_size
        self.num_legs = num_legs
        self.device = device
        self.reference_frame = reference_frame
        assert reference_frame in ["world", "base"], "reference_frame must be either 'world' or 'base'"
        
        # robot model
        self.biped = RobotFactory(robot)(batch_size, device)
        # swing leg trajectory
        self.swing_leg_trajectory = [SwingLegTrajectory(batch_size, device) for _ in range(num_legs)]
        
        # internal buffer
        self.state_estimator_data: StateEStimatorData = None
        self.desired_state_data: DesiredStateData = None
        self.leg_controller_data: LegControllerData = None

        self.swing_phase = torch.zeros((batch_size, self.num_legs), device=device) # swing subphase for each foot
        self.contact_phase = torch.zeros((batch_size, self.num_legs), device=device) # contact subphase for each foot
        self.swing_duration = swing_duration # (batch_size, num_legs) in seconds
        self.swing_time_remaining = self.swing_duration.clone() # (batch_size, num_legs) in seconds
        
        self.foot_height = swing_height * torch.ones(batch_size, device=device)
        self.cp1_coef = 1/3 * torch.ones(batch_size, device=device)
        self.cp2_coef = 2/3 * torch.ones(batch_size, device=device)
        self.stepping_frequency = 1.0
        
        # Foot placement variables
        self.foot_placement = torch.zeros((batch_size, num_legs, 3), device=device)
        self.foot_placement_b = torch.zeros((batch_size, num_legs, 3), device=device) # foot placement in body frame
        
        # First swing tracker
        self.first_swing = torch.ones((batch_size, num_legs), dtype=torch.bool, device=device)
        self.p0 = torch.zeros((batch_size, num_legs, 3), device=device) # foot position in body frame
    
    def set_state_estimator(self, data: StateEStimatorData):
        """
        Get com state and foot position in world
        """
        self.state_estimator_data = data
    
    def set_desired_state(self, data: DesiredStateData):
        """
        Get desired com state
        """
        self.desired_state_data = data

    def set_leg_controller_data(self, data: LegControllerData):
        """
        Get leg controller data
        """
        self.leg_controller_data = data
    
    def update_swing_phase(self, swing_phase: torch.Tensor):
        self.swing_phase = swing_phase

    def update_contact_phase(self, contact_phase: torch.Tensor):
        self.contact_phase = contact_phase

    def update_swing_duration(self, swing_duration: torch.Tensor):
        self.swing_duration = swing_duration
    
    def update_swing_time(self):
        """
        equivalent to updateswingtime in cpp
        """
        # Update swing time for each leg
        for leg in range(self.num_legs):
            first_swing_mask = self.first_swing[:, leg]
            
            # update remaining swing time
            self.swing_time_remaining[first_swing_mask, leg] = self.swing_duration[first_swing_mask, leg]
            self.swing_time_remaining[~first_swing_mask, leg] -= self.dt
            
            contact_mask = self.contact_phase[:, leg] >= 0
            self.first_swing[contact_mask, leg] = True
    
    def compute_foot_placement(self):
        """
        compute foot placement in global coordinate.
        use compute_raibert_heuristic_placement in foot_placement_planner.py
        equivalent to computeFootPlacement in cpp
        """
        # Get hip positions in body frame
        hip_positions = torch.zeros((self.batch_size, self.num_legs, 3), device=self.device)
        hip_positions[:, 0, :] = self.biped.hip_horizontal_location(0)
        hip_positions[:, 1, :] = self.biped.hip_horizontal_location(1)

        # Set swing trajectory parameters
        self.swing_leg_trajectory[0].set_height(self.foot_height)
        self.swing_leg_trajectory[1].set_height(self.foot_height)
        self.swing_leg_trajectory[0].set_control_points(self.cp1_coef, self.cp2_coef)
        self.swing_leg_trajectory[1].set_control_points(self.cp1_coef, self.cp2_coef)
        
        # Raibert heuristic parameters
        p_rel_max_x = 0.3
        p_rel_max_y = 0.3
        k_x = 0.03
        k_y = 0.03
        
        # Transform desired velocity from body to world frame
        v_des_robot = self.desired_state_data.desired_velocity_b
        v_des_world = (self.state_estimator_data.rotation_body @ v_des_robot.unsqueeze(2)).squeeze(2)
        
        # Compute foot placement using the Reipert heuristic
        self.foot_placement = compute_raibert_heuristic_placement(
            self.state_estimator_data.root_position,
            self.state_estimator_data.rotation_body,
            self.state_estimator_data.root_velocity_w,
            v_des_world,
            self.swing_duration,
            self.swing_time_remaining,
            p_rel_max_x,
            p_rel_max_y,
            k_x,
            k_y, 
            hip_positions,
        )

        # compute foot placement in base frame
        self.foot_placement_b[:, 0, :] = \
            (self.state_estimator_data.rotation_body.transpose(1, 2) @ (self.foot_placement[:, 0, :] - self.state_estimator_data.root_position).unsqueeze(2)).squeeze(2)
        
        self.foot_placement_b[:, 1, :] = \
            (self.state_estimator_data.rotation_body.transpose(1, 2) @ (self.foot_placement[:, 1, :] - self.state_estimator_data.root_position).unsqueeze(2)).squeeze(2)

        # Set landing location of swing leg trajectory
        for foot in range(self.num_legs):
            if self.reference_frame == "world":
                self.swing_leg_trajectory[foot].set_final_position(self.foot_placement[:, foot])
            elif self.reference_frame == "base":
                self.swing_leg_trajectory[foot].set_final_position(self.foot_placement_b[:, foot])
    
    def compute_foot_desired_position(self)->tuple[torch.Tensor, torch.Tensor]:
        """
        compute foot desired position in global coordinate.
        - get desired swing foot location in global coordinate
        - transform this location to body frame
        equivalent to computeFootDesiredPosition in cpp
        
        Returns:
            foot desired position in global coordinate [batch_size, num_legs, 3]
        """
        # Initialize desired foot positions and velocities in body frame
        p_foot_b = torch.zeros((self.batch_size, self.num_legs, 3), device=self.device)
        v_foot_b = torch.zeros((self.batch_size, self.num_legs, 3), device=self.device)
        
        for foot in range(self.num_legs): 
            # Create swing mask (for each batch element individually)
            first_swing_mask = torch.logical_and(self.first_swing[:, foot], self.swing_phase[:, foot] >= 0) # per foot
            
            # only update trajectory initial position when the foot is in first swing
            if self.reference_frame == "world":
                self.p0[first_swing_mask, foot, :] = self.state_estimator_data.foot_position[first_swing_mask, foot, :] # update buffer
                self.swing_leg_trajectory[foot].set_initial_position(self.p0[:, foot, :])
            elif self.reference_frame == "base":
                self.p0[first_swing_mask, foot, :] = self.leg_controller_data.p[first_swing_mask, foot, :] # update buffer
                self.swing_leg_trajectory[foot].set_initial_position(self.p0[:, foot, :])

            self.first_swing[self.swing_phase[:, foot] >=0, foot] = False
            self.first_swing[self.contact_phase[:, foot] >= 0, foot] = True
            
            # Compute trajectory for all batch elements in swing
            self.swing_leg_trajectory[foot].compute_swing_trajectory(
                self.swing_phase[:, foot],
                self.swing_duration[:, foot]
            )
            
            # Get position and velocity in world frame
            if self.reference_frame == "world":
                p_des_foot_world = self.swing_leg_trajectory[foot].get_position()
                v_des_foot_world = self.swing_leg_trajectory[foot].get_velocity()
                # transform desired foot state to body frame
                p_foot_b[:, foot] = (self.state_estimator_data.rotation_body.transpose(1, 2) @ (p_des_foot_world - self.state_estimator_data.root_position).unsqueeze(2)).squeeze(2)
                v_foot_b[:, foot] = (self.state_estimator_data.rotation_body.transpose(1, 2) @ (v_des_foot_world - self.state_estimator_data.root_velocity_w).unsqueeze(2)).squeeze(2)
            elif self.reference_frame == "base":
                p_foot_b[:, foot] = self.swing_leg_trajectory[foot].get_position()
                v_foot_b[:, foot] = self.swing_leg_trajectory[foot].get_velocity()

        return p_foot_b, v_foot_b
        
    def set_foot_height(self, foot_height: torch.Tensor):
        """
        Set the foot height
        """
        self.foot_height = foot_height

    def set_control_points(self, cp1: torch.Tensor, cp2: torch.Tensor):
        """
        Set the control points for the Bezier curve
        Args:
            cp1: coefficient for the first control point, typically between 0 and 1
            cp2: coefficient for the second control point, typically between 0 and 1
        """
        self.cp1_coef = cp1
        self.cp2_coef = cp2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import torch
    from state_estimator import StateEstimator
    
    dt = 0.001
    batch_size = 1 
    num_legs = 2
    device = torch.device("cpu")
    swing_duration = 0.2 * torch.ones((batch_size, num_legs), device=device)
    
    robot = "HECTOR"
    biped = RobotFactory(robot)(batch_size, device)
    state_est = StateEstimator(num_legs=num_legs, batch_size=batch_size, device=device)
    des_state = DesiredStateData(batch_size=batch_size, device=device)
    swing = SwingLegController(dt, batch_size, num_legs, device, swing_duration)
    
    
    # ====== set state estimation =====
    # Set root_position
    root_position = torch.zeros((batch_size, 3), device=device)
    root_position[:, 2] = 0.55  
    # Set root_quat to [0.7071068, 0, 0, 0.7071068] in w, x, y, z format
    root_quat = torch.zeros((batch_size, 4), device=device)
    root_quat[:, 0] = 1.0 
    # Set root velocity
    root_velocity_b = torch.zeros((batch_size, 3), device=device)
    root_velocity_b[:, 0] = 0.1
    # Set root angular velocity
    root_angular_velocity_b = torch.zeros((batch_size, 3), device=device)
    # Set the body state
    state_est.set_body_state(root_position, root_quat, root_velocity_b, root_angular_velocity_b)
    
    # Create foot_position_b
    foot_position_b = torch.zeros((batch_size, num_legs, 3), device=device)
    q = torch.zeros(batch_size, 5, device=device)
    q[:, 2] = torch.pi/4
    q[:, 3] = - torch.pi/2
    q[:, 4] = torch.pi/4
 
    biped.forward_kinematics_tree(q, 0)
    biped.forward_kinematics_tree(q, 1)
    foot_pos_left = biped.get_p0e(0)
    foot_pos_right = biped.get_p0e(1)

    foot_position_b[:, 0, :  ] = foot_pos_left
    foot_position_b[:, 1, :  ] = foot_pos_right
    
    # Update foot positions in state estimator
    state_est.update_foot_position(foot_position_b)
    # ====== /end set state estimation =====
    
    # === set desired state ===
    des_state.desired_velocity_b[:, 0] = 0.1
    
    # === set data to swing leg controller ===
    swing.set_state_estimator(state_est.data)
    swing.set_desired_state(des_state)
    
    # == simulate swing leg ===
    left_foot_positions_x = []
    left_foot_positions_y = []
    left_foot_positions_z = []
    
    right_foot_positions_x = []
    right_foot_positions_y = []
    right_foot_positions_z = []
    
    
    for swing_leg_index in [0, 1]:
        total_swing_phase = torch.linspace(0, 1, 100, device=device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_legs, 1)
        total_contact_phase = torch.linspace(0, 1, 100, device=device).unsqueeze(0).unsqueeze(1).repeat(batch_size, num_legs, 1)
        # right leg in swing, left leg in contact
        total_swing_phase[:, 1-swing_leg_index, :] = -1
        total_contact_phase[:, swing_leg_index, :] = -1
        for i in range(total_swing_phase.shape[2]):
            swing_phase = total_swing_phase[:, :, i]
            contact_phase = total_contact_phase[:, :, i]
            
            # Update controller with current phaseS
            swing.update_contact_phase(contact_phase)
            swing.update_swing_phase(swing_phase)
            swing.update_swing_time()
            swing.compute_foot_placement()
            
            # Get the desired foot position and velocity
            foot_position_b, foot_velocity_b = swing.compute_foot_desired_position_world()
            # Collect the position data for the first leg
            if swing_leg_index == 0:
                left_foot_positions_x.append(foot_position_b[0, swing_leg_index, 0].item())
                left_foot_positions_y.append(foot_position_b[0, swing_leg_index, 1].item())
                left_foot_positions_z.append(foot_position_b[0, swing_leg_index, 2].item())
            else:
                right_foot_positions_x.append(foot_position_b[0, swing_leg_index, 0].item())
                right_foot_positions_y.append(foot_position_b[0, swing_leg_index, 1].item())
                right_foot_positions_z.append(foot_position_b[0, swing_leg_index, 2].item())
        
    
    # Plot the figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(left_foot_positions_x,left_foot_positions_y, left_foot_positions_z, marker='o', linestyle='-', color="r")
    ax.plot(right_foot_positions_x,right_foot_positions_y, right_foot_positions_z, marker='o', linestyle='-', color="b")

    # Labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Foot Position Trajectory')
    ax.legend(['Left Foot', 'Right Foot'])
    
    # ax.set_aspect('equal', adjustable='box')
    # Show the plot
    plt.show()