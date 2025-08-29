from abc import ABC, abstractmethod
from typing import Union, Tuple
import torch

from biped_pympc.utils.math_utils import skew_symmetric
from biped_pympc.configuration.configuration import MPCConf
from biped_pympc.core.data.robot_data import StateEStimatorData, DesiredStateData, LegControllerData
from biped_pympc.core.robot.robot_factory import RobotFactory


class BaseMPCController(ABC):
    """
    Abstract base class for MPC controllers.
    
    This class defines the common interface that all MPC controller implementations
    must follow. Concrete implementations include:
    - MPCControllerCasadi
    - MPCControllerCusadi  
    - MPCControllerOSQP
    - MPCControllerQPSwift
    - MPCControllerQPTh
    """
    
    def __init__(self, num_envs: int, device: Union[torch.device, str], num_legs: int, cfg: MPCConf):
        """
        Initialize the base MPC controller.
        
        Args:
            num_envs: Number of environments/batch size
            device: Device to run computations on (CPU or GPU)
            num_legs: Number of legs on the robot
            cfg: MPC configuration object
        """
        self.cfg = cfg
        self.num_envs = num_envs
        self.num_legs = num_legs
        self.device = device
        self.horizon_length = cfg.horizon_length
        self.dt = cfg.dt
        self.dt_mpc = cfg.dt_mpc * torch.ones(num_envs, device=device)
        
        # Initialize dataclass
        self.state_estimate_data: StateEStimatorData = None
        self.desired_state_data: DesiredStateData = None
        self.leg_controller_data: LegControllerData = None
        
        # First run flag
        self.first_run = torch.ones(num_envs, device=device, dtype=torch.bool)
        
        # retrieve robot model
        self.biped = RobotFactory(cfg.robot)(num_envs, device)

        # Initialize buffers and solver
        self.init_buffer()
        self.init_solver()
    
    """
    Initialization
    """

    def init_buffer(self):
        nstate = 12
        self.R_body = torch.zeros(self.num_envs, self.horizon_length, 3, 3, device=self.device)
        self.I_world_inv = torch.zeros(self.num_envs, self.horizon_length, 3, 3, device=self.device)
        self.left_foot_pos_skew = torch.zeros(self.num_envs, self.horizon_length, 3, 3, device=self.device)
        self.right_foot_pos_skew = torch.zeros(self.num_envs, self.horizon_length, 3, 3, device=self.device)
        self.contact_table = torch.ones(self.num_envs, self.horizon_length, 2, device=self.device)
        self.x_ref = torch.zeros(self.num_envs, self.horizon_length, nstate, device=self.device)
        self.x0 = torch.zeros(self.num_envs, nstate, device=self.device)
        
        self.world_position_desired = torch.zeros(self.num_envs, 3, device=self.device)
        self.yaw_desired = torch.zeros(self.num_envs, device=self.device)
        
        self.mass = self.biped.mass
        self.mu = self.biped.mu
        
        self.Q = self.cfg.Q
        self.R = self.cfg.R

        self.residual_lin_accel = torch.zeros((self.num_envs, 3), device=self.device)
        self.residual_ang_accel = torch.zeros((self.num_envs, 3), device=self.device)

    
    @abstractmethod
    def init_solver(self) -> None:
        """
        Initialize the QP solver.
        
        This method should set up the specific QP solver implementation
        (OSQP, QPTh, CasADi, etc.) and load any necessary solver files.
        """
        pass

    """
    Setters
    """
    
    def set_state_estimate_data(self, state_estimate_data: StateEStimatorData) -> None:
        """
        Set the state estimate data for MPC computation.
        
        Args:
            state_estimate_data: Current state estimate from the robot
        """
        self.state_estimate_data = state_estimate_data
    
    def set_desired_state_data(self, desired_state_data: DesiredStateData) -> None:
        """
        Set the desired state data for MPC computation.
        
        Args:
            desired_state_data: Desired state from the high-level controller
        """
        self.desired_state_data = desired_state_data
    
    def set_leg_controller_data(self, leg_controller_data: LegControllerData) -> None:
        """
        Set the leg controller data for MPC computation.
        
        Args:
            leg_controller_data: Current leg controller state
        """
        self.leg_controller_data = leg_controller_data
    
    def set_contact_table(self, contact_table: torch.Tensor) -> None:
        """
        Set the contact table for the entire MPC horizon.
        
        Args:
            contact_table: Contact table indicating which feet are in contact
                          at each time step in the horizon
        """
        self.contact_table = contact_table.float()

    def set_mpc_sampling_time(self, dt_mpc: torch.Tensor) -> None:
        """
        Update MPC discretization timestep. 

        Args:
            dt_mpc (torch.Tensor): Tensor of shape (num_envs,) representing the new sampling time for each environment.
        """
        self.dt_mpc = dt_mpc
    

    """
    Operations.
    """
    
    @abstractmethod
    def run(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the MPC optimization.
        
        This method should:
        1. Compute knot points and horizon state
        2. Set initial state
        3. Compute reference trajectory
        4. Form and solve the QP problem
        5. Return the optimal forces and torques
        
        Returns:
            Tuple of (forces, torques) for each leg
        """
        pass
    
    def compute_knot_points(self) -> None:
        """
        Compute knot points for the MPC horizon.
        
        This method computes the robot state at each time step
        in the MPC horizon based on the current state and dynamics.
        """
        if self.first_run.any():
            self.world_position_desired[self.first_run, :] = self.state_estimate_data.root_position[self.first_run, :]
            self.yaw_desired[self.first_run] = self.state_estimate_data.root_euler[self.first_run, 2]
        self.first_run[self.first_run] = False
    
    def compute_horizon_state(self) -> None:
        """
        Compute the full state horizon for MPC.
        
        This method computes the complete state trajectory
        over the MPC horizon, including positions, velocities, and orientations.
        """
        # for now, use current state for the entire horizon
        # TODO: build integrator of COM dynamics and rollout during horizon
        self.R_body[:, :, :, :] = self.state_estimate_data.rotation_body.unsqueeze(1).repeat(1, self.horizon_length, 1, 1)
        self.I_world_inv[:, :, :, :] = torch.linalg.inv(
            self.state_estimate_data.rotation_body @ 
            self.biped.I_body.unsqueeze(0) @
            self.state_estimate_data.rotation_body.transpose(1, 2)
            ).unsqueeze(1).repeat(1, self.horizon_length, 1, 1)
        
        left_foot_rel = self.state_estimate_data.foot_position[:, 0, :] - self.state_estimate_data.root_position
        right_foot_rel = self.state_estimate_data.foot_position[:, 1, :] - self.state_estimate_data.root_position
        left_foot_rel_skew = skew_symmetric(left_foot_rel)
        right_foot_rel_skew = skew_symmetric(right_foot_rel)
        self.left_foot_pos_skew[:, :, :, :] = left_foot_rel_skew.unsqueeze(1).repeat(1, self.horizon_length, 1, 1)
        self.right_foot_pos_skew[:, :, :, :] = right_foot_rel_skew.unsqueeze(1).repeat(1, self.horizon_length, 1, 1)
    
    def set_initial_state(self) -> None:
        """
        Set the initial state for the MPC optimization.
        
        This method sets the current robot state as the initial
        condition for the MPC optimization.
        """
        self.x0[:, :3] = self.state_estimate_data.root_euler
        self.x0[:, 3:6] = self.state_estimate_data.root_position
        self.x0[:, 6:9] = self.state_estimate_data.root_angular_velocity_w
        self.x0[:, 9:12] = self.state_estimate_data.root_velocity_w
    
    def compute_reference_trajectory(self) -> None:
        """
        Get self.desired_state_data and 
        compute the reference trajectory for the entire horizon
        """
        time_bins = self.dt_mpc[:, None] * torch.arange(0, self.horizon_length, device=self.device).unsqueeze(0).repeat(self.num_envs, 1) # (0, dt_mpc, .., (h-1)*dt_mpc)
        
        # update open loop reference knot point
        self.world_position_desired[:, 0] += self.cfg.decimation * self.cfg.dt * self.desired_state_data.desired_velocity_b[:, 0]
        self.world_position_desired[:, 1] += self.cfg.decimation * self.cfg.dt * self.desired_state_data.desired_velocity_b[:, 1]
        self.world_position_desired[:, 2] = self.desired_state_data.desired_height
        self.yaw_desired += self.cfg.decimation * self.cfg.dt * self.desired_state_data.desired_angular_velocity_b[:, 2]

        stationary_mask = self.desired_state_data.desired_velocity_b[:, 0].abs() < 1e-2
        
        # rol, pitch, yaw
        self.x_ref[:, :, 0] = 0.0
        self.x_ref[:, :, 1] = 0.0
        # self.x_ref[:, :, 2] = 0.0
        # self.x_ref[:, :, 2] = self.state_estimate_data.root_euler[:, 2].unsqueeze(1) + self.desired_state_data.desired_angular_velocity_b[:, 2].unsqueeze(1) * time_bins
        self.x_ref[:, :, 2] = self.yaw_desired.unsqueeze(1) + self.desired_state_data.desired_angular_velocity_b[:, 2].unsqueeze(1) * time_bins
        
        # mask
        # if stationary_mask.any():
        #     self.x_ref[stationary_mask, :, 2] = self.yaw_desired[stationary_mask, None] + self.desired_state_data.desired_angular_velocity_b[stationary_mask, 2].unsqueeze(1) * time_bins
        
        # x, y, z
        desired_lin_velocity_w = (self.state_estimate_data.rotation_body @ self.desired_state_data.desired_velocity_b.unsqueeze(-1)).squeeze(-1)
        self.x_ref[:, :, 3] = self.state_estimate_data.root_position[:, 0].unsqueeze(1) + desired_lin_velocity_w[:, 0].unsqueeze(1) * time_bins
        self.x_ref[:, :, 4] = self.state_estimate_data.root_position[:, 1].unsqueeze(1) + desired_lin_velocity_w[:, 1].unsqueeze(1) * time_bins
        self.x_ref[:, :, 5] = self.desired_state_data.desired_height.unsqueeze(1).repeat(1, self.horizon_length)
        # mask
        # if stationary_mask.any():
        #     self.x_ref[stationary_mask, :, 3] = self.world_position_desired[stationary_mask, 0].unsqueeze(1) + desired_lin_velocity_w[stationary_mask, 0].unsqueeze(1) * time_bins
        #     self.x_ref[stationary_mask, :, 4] = self.world_position_desired[stationary_mask, 1].unsqueeze(1) + desired_lin_velocity_w[stationary_mask, 1].unsqueeze(1) * time_bins
        
        # wx, wy, wz
        self.x_ref[:, :, 6] = 0.0
        self.x_ref[:, :, 7] = 0.0
        self.x_ref[:, :, 8] = self.desired_state_data.desired_angular_velocity_b[:, 2].unsqueeze(1).repeat(1, self.horizon_length)
        
        # vx, vy, vz
        self.x_ref[:, :, 9] = desired_lin_velocity_w[:, 0].unsqueeze(1).repeat(1, self.horizon_length)
        self.x_ref[:, :, 10] = desired_lin_velocity_w[:, 1].unsqueeze(1).repeat(1, self.horizon_length)
        self.x_ref[:, :, 11] = 0.0

    def reset(self, env_ids:torch.Tensor):
        """
        Reset the MPC controller for a new episode.
        
        This method should reset the controller's internal state
        and prepare it for a new episode.
        """
        self.first_run[env_ids] = True