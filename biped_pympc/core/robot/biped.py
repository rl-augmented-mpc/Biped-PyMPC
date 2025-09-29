from dataclasses import dataclass, field
import torch

@dataclass
class DofCfg:
    kp: list[float] = field(default_factory=lambda: [])
    kd: list[float] = field(default_factory=lambda: [])
    torque_limit: list[float] = field(default_factory=lambda: [] )
    
class Biped:
    """Constants and functions related to the robot kinematics and dynamics."""
    
    def __init__(self, num_envs:int, device:torch.device):
        self.num_envs = num_envs
        self.device = device

        # pd conf
        self.pd_conf = DofCfg()
        
        self.define_dynamics_parameter()
        self.initialize_kinematics()
    
    def define_dynamics_parameter(self):
        self.mass = None
        self.I_body = None
        self.mu = None
        
    @property
    def num_dof(self):
        return 5
    
    """
    Kinematics code
    """
    def hip_horizontal_location(self, leg:int)->torch.Tensor:
        raise NotImplementedError
    
    def initialize_kinematics(self):
        """
        Define kinematic coordinate system for each leg.
        If manually defined, this function initializes all tensors required for forward kinematics.
        If using tool like pinocchio, URDF is parsed and model and data are initialized.
        """
        raise NotImplementedError
    
    def forward_kinematics(self, joint_angle: torch.Tensor, leg:int)->None:
        """
        Compute forward kinematics of each leg given joint angles.
        """
        raise NotImplementedError
    
    # call this after forward_kinematics_tree
    def get_p0e(self, leg:int)->torch.Tensor:
        """
        retrieve foot end effector position wrt base frame
        """
        raise NotImplementedError
    
    def contact_jacobian(self, leg:int)->torch.Tensor:
        """
        get contact Jacobian given current forward kinematics results.
        
        Args:
            leg (int): Leg identifier (0 for left, 1 for right).
        Returns:
            torch.Tensor: Contact Jacobian (batch_size, 6, num_dof)
        """
        raise NotImplementedError
         
    def analytical_IK(self, p_foot_des_b:torch.Tensor, leg:int)->torch.Tensor:
        """
        Compute the inverse kinematics for a bipedal robot leg with batch support.
        Since we only track 3D position, there can be only 3 variables to solve for.
        This is why we fix yaw angle to be 0 and ankle angle to be aligned with the torso pitch.
        
        Args:
            p_hip2foot (torch.Tensor): Desired foot position in hip frame (batch_size x 3).
            leg (int): Leg identifier (0 for left, 1 for right).

        Returns:
            torch.Tensor: Joint angles (batch_size, 5)
        """
        raise NotImplementedError