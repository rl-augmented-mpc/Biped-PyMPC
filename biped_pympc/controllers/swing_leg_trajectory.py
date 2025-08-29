from typing import Literal
import torch

class SwingLegTrajectory:
    def __init__(self, 
                 batch_size: int, 
                 device: torch.device = torch.device("cpu"), 
                 curve_type: Literal["cycloid", "bezier"] = "bezier"):
        """
        Constructor setting everything to zero
        """
        self.batch_size = batch_size
        self.device = device
        self.p0 = torch.zeros((batch_size, 3), device=self.device)
        self.pf = torch.zeros((batch_size, 3), device=self.device)
        self.p = torch.zeros((batch_size, 3), device=self.device)
        self.v = torch.zeros((batch_size, 3), device=self.device)
        self.height = torch.zeros(batch_size, device=self.device)
        
        self.curve_type = curve_type
        self.cp1 = 1/3 * torch.ones(batch_size, device=self.device)
        self.cp2 = 2/3 * torch.ones(batch_size, device=self.device)

    def set_initial_position(self, p0: torch.Tensor):
        """
        Set the starting position of the foot
        Args:
            p0: the initial position of the foot
        """
        self.p0 = p0

    def set_final_position(self, pf: torch.Tensor):
        """
        Set the desired final position of the foot
        Args:
            pf: the final position of the foot
        """
        self.pf = pf

    def set_height(self, height: float):
        """
        Set the maximum height of the swing
        Args:
            height: the maximum height of the swing, achieved halfway through the swing
        """
        self.height = height * torch.ones(self.batch_size, device=self.device)

    def set_control_points(self, cp1: torch.Tensor, cp2: torch.Tensor):
        """
        Set the control points for the Bezier curve
        Args:
            cp1: coefficient for the first control point, typically between 0 and 1
            cp2: coefficient for the second control point, typically between 0 and 1
        """
        self.cp1 = cp1
        self.cp2 = cp2

    def get_position(self):
        """
        Get the current foot position
        Returns:
            the current foot position
        """
        return self.p
    
    def get_initial_position(self):
        """
        Get the initial foot position
        Returns:
            the initial foot position
        """
        return self.p0

    def get_velocity(self):
        """
        Get the current foot velocity
        Returns:
            the current foot velocity
        """
        return self.v
    
    def compute_swing_trajectory(self, phase:torch.Tensor, swing_time:torch.Tensor):
        """
        Compute the foot position and velocity using a Bezier curve.

        Args:
            phase: Current phase of the swing (0 to 1).
            swing_time: Total time of the swing in seconds.
        """
        
        if self.curve_type == "bezier":
            self.p, self.v = self.cubic_bezier_trajectory(phase, swing_time, self.p0, self.pf, self.height, self.cp1, self.cp2)
            
        elif self.curve_type == "cycloid":
            self.p, self.v = self.cycloid_trajectory(phase, swing_time, self.p0, self.pf, self.height)
        
    @torch.jit.script
    def cubic_bezier_trajectory(
        phase:torch.Tensor, 
        swing_time:torch.Tensor, 
        p0:torch.Tensor,
        pf:torch.Tensor,
        height:torch.Tensor, 
        cp1_coef:torch.Tensor,
        cp2_coef:torch.Tensor,
    ):
        """
        Compute the foot position and velocity using a cubic Bezier trajectory.

        Args:
            phase: Current phase of the swing (0 to 1).
            swing_time: Total time of the swing in seconds.
            p0: Initial position of the foot.
            pf: Final position of the foot.
            height: Maximum height of the swing.
            cp1_coef: Coefficient for the first control point.
            cp2_coef: Coefficient for the second control point.
        """
        p1 = p0 + cp1_coef[:, None] * (pf - p0)
        p2 = p0 + cp2_coef[:, None] * (pf - p0)
        z_apex = p0[:, 2] + height
        
        # compute control point height to achieve desired apex height (assumption: peak at phase=0.5)
        p1[:, 2] = (8 * z_apex - p0[:, 2] - pf[:, 2]) / 6.0
        p2[:, 2] = (8 * z_apex - p0[:, 2] - pf[:, 2]) / 6.0
        
        p = (1-phase).pow(3)[:, None] * p0 + \
            3 * (1-phase).pow(2)[:, None] * phase[:, None] * p1 + \
            3 * (1-phase)[:, None] * phase.pow(2)[:, None] * p2 + \
            phase.pow(3)[:, None] * pf
            
        v = (3 * (1-phase).pow(2)[:, None] * (p1 - p0) + \
            6 * (1-phase)[:, None] * phase[:, None] * (p2 - p1) + \
            3 * phase.pow(2)[:, None] * (pf - p2)) / swing_time[:, None]
        
        return p, v
    
    
    @torch.jit.script
    def cycloid_trajectory(
        phase:torch.Tensor, 
        swing_time:torch.Tensor, 
        p0:torch.Tensor,
        pf:torch.Tensor,
        height:torch.Tensor
    ):
        """
        Compute the foot position and velocity using a cycloid trajectory.

        Args:
            phase: Current phase of the swing (0 to 1).
            swing_time: Total time of the swing in seconds.
            p0: Initial position of the foot.
            pf: Final position of the foot.
            height: Maximum height of the swing.
        """
        phase_pi = 2 * torch.pi * phase
        p = (pf - p0) * (phase_pi[:, None] - torch.sin(phase_pi[:, None])) / (2 * torch.pi) + p0
        v = (pf - p0) * (1 - torch.cos(phase_pi[:, None])) / swing_time[:, None]
        p[:, 2] = height * (1 - torch.cos(phase_pi)) / 2.0 + p0[:, 2]
        v[:, 2] = height * torch.pi * torch.sin(phase_pi) / swing_time
        
        return p, v