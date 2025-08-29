import torch
from dataclasses import dataclass

"""
stack of dataclass for managing data
"""

@dataclass
class StateEStimatorData:
    num_legs: int = 2
    batch_size: int = 1
    device: torch.device = torch.device("cpu")
    
    def __post_init__(self):
        # in global odometry frame (sort of like world frame)
        self.root_position = torch.zeros((self.batch_size, 3), device=self.device)
        self.root_quat = torch.zeros((self.batch_size, 4), device=self.device)
        self.root_quat[:, 0] = 1.0
        self.root_euler = torch.zeros((self.batch_size, 3), device=self.device)
        self.rotation_body = torch.eye(3, device=self.device).unsqueeze(0).repeat(self.batch_size, 1, 1)
        self.root_velocity_w = torch.zeros((self.batch_size, 3), device=self.device)
        self.root_angular_velocity_w = torch.zeros((self.batch_size, 3), device=self.device)
        self.foot_position = torch.zeros((self.batch_size, self.num_legs, 3), device=self.device)
        
        # in local frame
        self.root_velocity_b = torch.zeros((self.batch_size, 3), device=self.device)
        self.root_angular_velocity_b = torch.zeros((self.batch_size, 3), device=self.device)

@dataclass
class DesiredStateData:
    """
    Desired state in body frame
    """
    batch_size: int = 1
    device: torch.device = torch.device("cpu")
    
    def __post_init__(self):
        self.desired_velocity_b = torch.zeros((self.batch_size, 3), device=self.device)
        self.desired_angular_velocity_b = torch.zeros((self.batch_size, 3), device=self.device)
        self.desired_height = 0.55 * torch.ones(self.batch_size, device=self.device)
        
        # TODO: not sure if this is needed
        self.desired_position = torch.zeros((self.batch_size, 3), device=self.device)
        self.desired_angle = torch.zeros((self.batch_size, 3), device=self.device) # roll, pitch, yaw
    
    def set_command(self, desired_lin_velocity:torch.Tensor, desired_ang_velocity:torch.Tensor, desired_height:torch.Tensor)->None:
        """
        desired_lin_velocity: (batch_size, 2)
        desired_ang_velocity: (batch_size, )
        desired_height: (batch_size, )
        """
        self.desired_velocity_b[:, :2] = desired_lin_velocity
        self.desired_angular_velocity_b[:, 2] = desired_ang_velocity
        self.desired_height[:] = desired_height

@dataclass
class LegControllerCommand:
    num_leg: int = 2
    num_dof: int = 5
    batch_size: int = 1
    device: torch.device = torch.device("cpu")
    
    def __post_init__(self):
        self.tau = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)
        self.qDes = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)
        self.qdDes = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)

        self.pDes = torch.zeros(self.batch_size, self.num_leg, 3, device=self.device)
        self.vDes = torch.zeros(self.batch_size, self.num_leg, 3, device=self.device)
        self.feedfowardforce = torch.zeros(self.batch_size, self.num_leg, 6, device=self.device)

        self.kpjoint = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)
        self.kdjoint = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)
        
        # residuals
        self.qDesDelta = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)
        self.feedfowardforceDelta = torch.zeros(self.batch_size, self.num_leg, 6, device=self.device)
        self.footplacementDelta = torch.zeros(self.batch_size, self.num_leg, 2, device=self.device)
        self.Pf = torch.zeros(self.batch_size, self.num_leg, 3, device=self.device)
        self.Pf_aug = torch.zeros(self.batch_size, self.num_leg, 3, device=self.device)

@dataclass
class LegControllerData:
    num_leg: int = 2
    num_dof: int = 5
    batch_size: int = 1
    device: torch.device = torch.device("cpu")
    
    def __post_init__(self):
        self.tau = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)
        self.q = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)
        self.qd = torch.zeros(self.batch_size, self.num_leg, self.num_dof, device=self.device)

        self.J = torch.zeros(self.batch_size, self.num_leg, 6, self.num_dof, device=self.device)
        self.Jv = torch.zeros(self.batch_size, self.num_leg, 3, self.num_dof, device=self.device)

        self.p = torch.zeros(self.batch_size, self.num_leg, 3, device=self.device)
        self.v = torch.zeros(self.batch_size, self.num_leg, 3, device=self.device)
        
        # contact states
        self.contact_phase = torch.zeros(self.batch_size, self.num_leg, device=self.device)
        self.contact_bool = torch.ones(self.batch_size, self.num_leg, device=self.device)
        self.swing_phase = torch.zeros(self.batch_size, self.num_leg, device=self.device)
        self.swing_bool = torch.zeros(self.batch_size, self.num_leg, device=self.device)