from typing import Union
import torch

from biped_pympc.core.data.robot_data import StateEStimatorData
from biped_pympc.utils.math_utils import quaternion_to_rotation_matrix, quat_to_euler
    
class StateEstimator:
    """
    State estimator interface. 
    It takes state estimator results (which comes from some external module) and
    provides an interface to access the state data. 
    
    State data includes 
    - center of mass state in global frame (or odometry frame)
    - center of mass state in its body frame
    - foot position in global frame
    """
    def __init__(self, num_legs:int, batch_size:int, device:Union[torch.device, str] = torch.device("cpu")):
        self.num_legs = num_legs
        self.data = StateEStimatorData(num_legs=num_legs, batch_size=batch_size, device=device)
    
    def set_body_state(self, 
                       root_position:torch.Tensor, 
                       root_quat:torch.Tensor, 
                       root_velocity_b:torch.Tensor, 
                       root_angular_velocity_b:torch.Tensor)->None:
        # set the body state available from state estimator module
        self.data.root_position = root_position
        self.data.root_quat = root_quat
        self.data.root_velocity_b = root_velocity_b
        self.data.root_angular_velocity_b = root_angular_velocity_b
        
        # compute euler angles from quaternion
        self.data.root_euler = quat_to_euler(root_quat)
        # compute the rotation matrix from quaternion
        self.data.rotation_body = quaternion_to_rotation_matrix(root_quat)
        
        self.data.root_velocity_w = torch.bmm(self.data.rotation_body, root_velocity_b.unsqueeze(-1)).squeeze(-1)
        self.data.root_angular_velocity_w = torch.bmm(self.data.rotation_body, root_angular_velocity_b.unsqueeze(-1)).squeeze(-1)
    
    def update_foot_position(self, foot_position_b:torch.Tensor)->None:
        # update foot position in global frame
        rot = self.data.rotation_body.unsqueeze(1).repeat(1, self.data.num_legs, 1, 1).reshape(-1, 3, 3)
        self.data.foot_position = torch.bmm(rot, foot_position_b.reshape(-1, 3, 1)).squeeze(-1).reshape(-1, self.num_legs, 3) \
            + self.data.root_position.unsqueeze(1)


if __name__ == "__main__":
    num_legs = 2
    batch_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    state_estimator = StateEstimator(num_legs=num_legs, batch_size=batch_size, device=device)
    root_position = torch.zeros((batch_size, 3), device=device)
    root_position[:, 2] = 0.55
    root_quat = torch.zeros((batch_size, 4), device=device)
    root_quat[:, 0] = 1.0
    root_velocity_b = torch.zeros((batch_size, 3), device=device)
    root_angular_velocity_b = torch.zeros((batch_size, 3), device=device)
    foot_position_b = torch.zeros((batch_size, num_legs, 3), device=device)
    foot_position_b[:, 0, 0] = 0.1
    foot_position_b[:, 1, 0] = 0.1
    foot_position_b[:, 0, 1] = 0.05
    foot_position_b[:, 1, 1] = -0.05
    foot_position_b[:, 0, 2] = -0.55
    foot_position_b[:, 1, 2] = -0.55
    
    state_estimator.set_body_state(root_position, root_quat, root_velocity_b, root_angular_velocity_b)
    state_estimator.update_foot_position(foot_position_b)
    
    print(state_estimator.data.foot_position[0])
    print(state_estimator.data.rotation_body[0])