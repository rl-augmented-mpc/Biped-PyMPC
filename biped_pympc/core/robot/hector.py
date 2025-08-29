from dataclasses import dataclass, field
import torch
from biped_pympc.core.robot.biped import Biped
from biped_pympc.utils.math_utils import rot_x, rot_z

@dataclass
class DofCfg:
    torque_limit: list[float] = field(default_factory=lambda: [33.5, 33.5, 33.5, 67.0, 33.5, 33.5, 33.5, 33.5, 67.0, 33.5])
    # kp: list[float] = field(default_factory=lambda: [20.0, 20.0, 20.0, 20.0, 15.0])
    # kd: list[float] = field(default_factory=lambda: [0.45, 0.45, 0.45, 0.45, 0.6])
    kp: list[float] = field(default_factory=lambda: [40.0, 40.0, 70.0, 70.0, 40.0])
    kd: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.7, 0.7, 0.7])

class HECTOR(Biped):
    """Constants and functions related to the robot kinematics and dynamics."""
    
    def __init__(self, num_envs:int, device:torch.device):
        self.num_envs = num_envs
        self.device = device
        
        # pd conf
        self.pd_conf = DofCfg()
        
        # left leg hip yaw configuration
        self.hip_yaw = torch.tensor([-0.00, 0.047, -0.126], device=self.device) # hip yaw joint position wrt torso frame
        self.hip_roll = torch.tensor([0.0465, 0.015, -0.0705], device=self.device) # hip roll joint position wrt hip yaw frame
        
        self.define_dynamics_parameter()
        self.initialize_kinematics()
    
    def define_dynamics_parameter(self):
        self.mass = 13.856
        self.I_body = torch.tensor([[0.5413, 0.0, 0.0],
                                    [0.0, 0.5200, 0.0],
                                    [0.0, 0.0, 0.0691]], device=self.device)
        self.mu = 1.0
        
    @property
    def num_dof(self):
        return 5
    
    """
    Kinematics code
    """
    def hip_horizontal_location(self, leg:int)->torch.Tensor:
        side = 1*(1-leg) -1*leg # leg0: side=1, leg1: side=-1
        hip_roll_location = torch.zeros(self.num_envs, 3, device=self.device)
        hip_roll_location[:, 0] = self.hip_yaw[0] + self.hip_roll[0] - 0.06
        hip_roll_location[:, 1] = side * (self.hip_yaw[1] + self.hip_roll[1]+0.036)
        return hip_roll_location
    
    def initialize_kinematics(self):
        # left leg link configurations        
        self.p1 = torch.tensor([-0.00, 0.047, -0.1265], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.p2 = torch.tensor([0.0465, 0.015, -0.0705], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.p3 = torch.tensor([-0.06, 0.018, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.p4 = torch.tensor([0.0, 0.01805, -0.22], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.p5 = torch.tensor([0.0, 0.00, -0.22], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.p5e = torch.tensor([0.0, 0.0, -0.042, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        self.z5 = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.z4 = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.z3 = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.z2 = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.z1 = torch.tensor([0.0, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        
        self.T45_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T34_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T3p3_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T23p_left = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], 
                                      dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T2p2_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T12p_left = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], 
                                      dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T01_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        self.T45_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T34_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T3p3_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T23p_right = torch.tensor([[0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]], 
                                       dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T2p2_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T12p_right = torch.tensor([[0, 0, 1, 0], [0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 0, 1]], 
                                       dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T01_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        # body to each link transform 
        self.T02_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T02_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T03_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T03_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T04_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T04_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T05_left = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.T05_right = torch.eye(4, device=self.device).unsqueeze(0).repeat(self.num_envs, 1, 1)
        
        self.p0e_left = torch.zeros(self.num_envs, 3, device=self.device)
        self.p0e_right = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.J_left = torch.zeros(self.num_envs, 6, 5, device=self.device)
        self.J_right = torch.zeros(self.num_envs, 6, 5, device=self.device)
        
        # rotate offset vectors
        self.p2 = (self.T12p_left[:, :3, :3].transpose(1, 2) @ self.p2.unsqueeze(2)).squeeze(2)
        self.p3 = (self.T23p_left[:, :3, :3].transpose(1, 2) @ self.T12p_left[:, :3, :3].transpose(1, 2) @ self.p3.unsqueeze(2)).squeeze(2)
        self.p4 = (self.T23p_left[:, :3, :3].transpose(1, 2) @ self.T12p_left[:, :3, :3].transpose(1, 2) @ self.p4.unsqueeze(2)).squeeze(2)
        self.p5 = (self.T23p_left[:, :3, :3].transpose(1, 2) @ self.T12p_left[:, :3, :3].transpose(1, 2) @ self.p5.unsqueeze(2)).squeeze(2)
        self.p5e = (self.T23p_left.transpose(1, 2) @ self.T12p_left.transpose(1, 2) @ self.p5e.unsqueeze(2)).squeeze(2)
    
    def forward_kinematics(self, joint_angle: torch.Tensor, leg:int)->None:
        if leg == 0:
            self.T45_left[:, :3, :3] = rot_z(joint_angle[:, 4])
            self.T45_left[:, :3, 3] = self.p5
            
            self.T34_left[:, :3, :3] = rot_z(joint_angle[:, 3])
            self.T34_left[:, :3, 3] = self.p4
            
            self.T3p3_left[:, :3, :3] = rot_z(joint_angle[:, 2])
            self.T3p3_left[:, :3, 3] = self.p3
            
            self.T2p2_left[:, :3, :3] = rot_z(joint_angle[:, 1])
            self.T2p2_left[:, :3, 3] = self.p2
            
            self.T01_left[:, :3, :3] = rot_z(joint_angle[:, 0])
            self.T01_left[:, :3, 3] = self.p1
            
            self.T02_left[:, :, :] = self.T01_left @ self.T12p_left @ self.T2p2_left
            self.T03_left[:, :, :] = self.T02_left @ self.T23p_left @ self.T3p3_left
            self.T04_left[:, :, :] = self.T03_left @ self.T34_left
            self.T05_left[:, :, :] = self.T04_left @ self.T45_left
            
            # foot sole position in ankle frame
            self.p0e_left[:, :] = (self.T05_left @ self.p5e.unsqueeze(2)).squeeze(2)[:, :3]
        else:
            self.T45_right[:, :3, :3] = rot_z(joint_angle[:, 4])
            self.T45_right[:, :3, 3] = self.p5 * torch.tensor([1, 1, -1], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            
            self.T34_right[:, :3, :3] = rot_z(joint_angle[:, 3])
            self.T34_right[:, :3, 3] = self.p4 * torch.tensor([1, 1, -1], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            
            self.T3p3_right[:, :3, :3] = rot_z(joint_angle[:, 2])
            self.T3p3_right[:, :3, 3] = self.p3 * torch.tensor([1, 1, -1], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            
            self.T2p2_right[:, :3, :3] = rot_z(joint_angle[:, 1])
            self.T2p2_right[:, :3, 3] = self.p2 * torch.tensor([1, -1, 1], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            
            self.T01_right[:, :3, :3] = rot_z(joint_angle[:, 0])
            self.T01_right[:, :3, 3] = self.p1 * torch.tensor([1, -1, 1], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            
            self.T02_right[:, :, :] = self.T01_right @ self.T12p_right @ self.T2p2_right
            self.T03_right[:, :, :] = self.T02_right @ self.T23p_right @ self.T3p3_right
            self.T04_right[:, :, :] = self.T03_right @ self.T34_right
            self.T05_right[:, :, :] = self.T04_right @ self.T45_right
            
            # foot sole position in ankle frame
            self.p0e_right[:, :] = (self.T05_right @ self.p5e.unsqueeze(2)).squeeze(2)[:, :3]
    
    # call this after forward_kinematics_tree
    def get_p0e(self, leg:int)->torch.Tensor:
        """
        retrieve foot end effector position wrt base frame
        """
        if leg == 0:
            return self.p0e_left
        else:
            return self.p0e_right
    
    def contact_jacobian(self, leg:int)->torch.Tensor:
        if leg == 0:
            p01 = self.T01_left[:, :3, 3]
            p02 = self.T02_left[:,:3, 3]
            p03 = self.T03_left[:,:3, 3]
            p04 = self.T04_left[:,:3, 3]
            p05 = self.T05_left[:,:3, 3]
            
            z1 = (self.T01_left[:, :3, :3] @ self.z1.unsqueeze(2)).squeeze(2)
            z2 = (self.T02_left[:, :3, :3] @ self.z2.unsqueeze(2)).squeeze(2)
            z3 = (self.T03_left[:, :3, :3] @ self.z3.unsqueeze(2)).squeeze(2)
            z4 = (self.T04_left[:, :3, :3] @ self.z4.unsqueeze(2)).squeeze(2)
            z5 = (self.T05_left[:, :3, :3] @ self.z5.unsqueeze(2)).squeeze(2)
            
            self.J_left[:, :3, 0] = torch.cross(z1, (self.p0e_left - p01), dim=1)
            self.J_left[:, 3:, 0] = z1
            self.J_left[:, :3, 1] = torch.cross(z2, (self.p0e_left - p02), dim=1)
            self.J_left[:, 3:, 1] = z2
            self.J_left[:, :3, 2] = torch.cross(z3, (self.p0e_left - p03), dim=1)
            self.J_left[:, 3:, 2] = z3
            self.J_left[:, :3, 3] = torch.cross(z4, (self.p0e_left - p04), dim=1)
            self.J_left[:, 3:, 3] = z4
            self.J_left[:, :3, 4] = torch.cross(z5, (self.p0e_left - p05), dim=1)
            self.J_left[:, 3:, 4] = z5
            return self.J_left
        else:
            p01 = self.T01_right[:, :3, 3]
            p02 = self.T02_right[:,:3, 3]
            p03 = self.T03_right[:,:3, 3]
            p04 = self.T04_right[:,:3, 3]
            p05 = self.T05_right[:,:3, 3]
            
            z1 = (self.T01_left[:, :3, :3] @ self.z1.unsqueeze(2)).squeeze(2)
            z2 = (self.T02_left[:, :3, :3] @ self.z2.unsqueeze(2)).squeeze(2)
            z3 = (self.T03_left[:, :3, :3] @ self.z3.unsqueeze(2)).squeeze(2)
            z4 = (self.T04_left[:, :3, :3] @ self.z4.unsqueeze(2)).squeeze(2)
            z5 = (self.T05_left[:, :3, :3] @ self.z5.unsqueeze(2)).squeeze(2)
            
            self.J_right[:, :3, 0] = torch.cross(z1, (self.p0e_right - p01), dim=1)
            self.J_right[:, 3:, 0] = z1
            self.J_right[:, :3, 1] = torch.cross(z2, (self.p0e_right - p02), dim=1)
            self.J_right[:, 3:, 1] = z2
            self.J_right[:, :3, 2] = torch.cross(z3, (self.p0e_right - p03), dim=1)
            self.J_right[:, 3:, 2] = z3
            self.J_right[:, :3, 3] = torch.cross(z4, (self.p0e_right - p04), dim=1)
            self.J_right[:, 3:, 3] = z4
            self.J_right[:, :3, 4] = torch.cross(z5, (self.p0e_right - p05), dim=1)
            self.J_right[:, 3:, 4] = z5
            return self.J_right
         
    def analytical_IK(self, p_foot_des_b:torch.Tensor, leg:int)->torch.Tensor:
        """
        Compute the inverse kinematics for a bipedal robot leg with batch support.
        Since we only track 3D position, there can be only 3 variables to solve for.
        This is why we fix yaw angle to be 0 and ankle angle to be aligned with the torso pitch.
        
        Args:
            p_foot_des_b (torch.Tensor): Desired foot position in hip frame (batch_size x 3).
            leg (int): Leg identifier (0 for left, 1 for right).

        Returns:
            torch.Tensor: Joint angles (batch_size, num_dof)
        """
        side = 1*leg -1*(1-leg) # -1: for left leg, 1: for right leg
        
        # torso to hip roll offset
        torso_hip_roll_offset = torch.zeros(p_foot_des_b.shape[0], 3, device=p_foot_des_b.device)
        torso_hip_roll_offset[:, 0] = -0.00+0.0465-0.06
        torso_hip_roll_offset[:, 1] = -side * (0.047+0.015)
        torso_hip_roll_offset[:, 2] = -0.126-0.0705
        
        # Compute desired foot position in the hip roll frame
        foot_des_from_hip_roll = p_foot_des_b - torso_hip_roll_offset
        
        thigh_length = 0.22
        calf_length = 0.22
        d_foot = 0.042
        foot_des_from_hip_roll[:, 2] += d_foot

        # Distances
        distance_2D_yOz = torch.sqrt(foot_des_from_hip_roll[:, 1]**2 + foot_des_from_hip_roll[:, 2]**2)
        distance_horizontal = 0.018+0.01805 # hip roll to ankle y distance
        
        q = torch.zeros(self.num_envs, self.num_dof, device=p_foot_des_b.device)  # Joint angles
        
        # Joint angle computations
        
        # hip roll 
        q[:, 1] = torch.asin(torch.clamp(foot_des_from_hip_roll[:, 1] / distance_2D_yOz, -1.0, 1.0)) + \
                torch.asin(torch.clamp(distance_horizontal * side / distance_2D_yOz, -1.0, 1.0))
                
        # transform foot_des_from_hip_roll from base frame to hip pitch frame
        R_hip_roll = rot_x(q[:, 1])
        hip_roll_to_hip_pitch_offset = torch.zeros(self.num_envs, 3, device=self.device)
        hip_roll_to_hip_pitch_offset[:, 1] = 0.018*side
        foot_des_from_hip_pitch = (R_hip_roll @ foot_des_from_hip_roll.unsqueeze(2)).squeeze(2) + hip_roll_to_hip_pitch_offset
        r = torch.norm(foot_des_from_hip_pitch, dim=1)
        
        # planar 2R IK in x-z plane
        cos_q2 = torch.clamp((torch.square(r) - thigh_length**2 - calf_length**2) / (2.0 * thigh_length * calf_length), -1.0, 1.0)
        sin_q2 = torch.clamp(-torch.sqrt(torch.clamp(1.0 - torch.square(cos_q2), min=1e-6)), -1.0, 1.0)
        q[:, 3] = torch.atan2(sin_q2, cos_q2)  # knee angle
        q[:, 2] = torch.atan2(-foot_des_from_hip_pitch[:, 0], -foot_des_from_hip_pitch[:, 2]) - \
            torch.atan2(calf_length * sin_q2, thigh_length + calf_length * cos_q2)  # hip pitch angle
        q[:, 4] = -q[:, 2] - q[:, 3]  # ankle angle

        return q
    
    # just for debugging
    def get_p1e(self, leg:int)->torch.Tensor:
        """
        retrieve foot-hip to end-effector (foot sole) displacement wrt torso frame
        """
        if leg == 0:
            return self.p0e_left - self.p1
        else:
            return self.p0e_right - (self.p1 * torch.tensor([1, -1, 1], device=self.device).unsqueeze(0).repeat(self.num_envs, 1))
    
    def get_p0toe(self, leg:int)->torch.Tensor:
        """
        retrieve foot toe position wrt torso frame
        """
        p5e = torch.tensor([0.07, 0.04, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        if leg == 0:
            return (self.T05_left @ p5e.unsqueeze(2)).squeeze(2)[:, :3]
        else:
            return (self.T05_right @ p5e.unsqueeze(2)).squeeze(2)[:, :3]
    
    def get_p0heel(self, leg:int)->torch.Tensor:
        """
        retrieve foot heel position wrt torso frame
        """
        p5e = torch.tensor([-0.04, 0.04, 0.0, 1.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        if leg == 0:
            return (self.T05_left @ p5e.unsqueeze(2)).squeeze(2)[:, :3]
        else:
            return (self.T05_right @ p5e.unsqueeze(2)).squeeze(2)[:, :3]

if __name__ == "__main__":
    import numpy as np
    import timeit
    torch.manual_seed(0)
    num_envs = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    biped = HECTOR(num_envs, device)
    
    # initialize joint angle
    q = torch.zeros(num_envs, 5, device=device)
    q[:, 2] = torch.pi/4
    q[:, 3] = - torch.pi/2
    q[:, 4] = torch.pi/4
    print("given joint angle", (180/torch.pi)*q[0])
    
    # test FK
    print("FK")
    biped.forward_kinematics_tree(q, 0)
    biped.forward_kinematics_tree(q, 1)
    foot_pos_left = biped.get_p0e(0)
    foot_pos_right = biped.get_p0e(1)
    J_left = biped.contact_jacobian(0)
    J_right = biped.contact_jacobian(1)
    
    print("foot_pos_left", foot_pos_left[0])
    print("foot_pos_right", foot_pos_right[0])
    
    # test IK #
    foot_des = foot_pos_left.clone()
    joint_angle_left = biped.analytical_IK(foot_des, 0)
    print("joint_angle_left", (180/torch.pi)*joint_angle_left[0])
    
    foot_des = foot_pos_right.clone()
    joint_angle_right = biped.analytical_IK(foot_des, 1)
    print("joint_angle_right", (180/torch.pi)*joint_angle_right[0])
    
    # ################################
    # # === Performance evaluation ===
    # number = 10
    # # FK evaluation time
    # execution_time = timeit.timeit(lambda: biped.forward_kinematics(q, 0), number=number)
    # print(f"Old FK exec time: {1000*(execution_time/number):.6f} ms")
    
    # execution_time = timeit.timeit(lambda: biped.forward_kinematics_tree(q, 0), number=number)
    # print(f"New FK exec time: {1000*(execution_time/number):.6f} ms")
    
    # # jacobian evaluation time
    # execution_time = timeit.timeit(lambda: biped.foot_jacobian(q, 0), number=number)
    # print(f"Old Inverse dynamics exec time: {1000*(execution_time/number):.6f} ms")
    
    # execution_time = timeit.timeit(lambda: biped.contact_jacobian(0), number=number)
    # print(f"New Inverse dynamics exec time: {1000*(execution_time/number):.6f} ms")
    
    # # IK evaluation time
    # execution_time = timeit.timeit(lambda: biped.inverse_kinematics(foot_des, 0), number=number)
    # print(f"IK exec time: {1000*(execution_time/number):.6f} ms")
    
    
    # #########################################
    # # === Debug FK/IK with visualization ===
    # import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')
    
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    
    # q = torch.zeros(num_envs, 5, device=device)
    # q[:, 2] = torch.pi/4
    # q[:, 3] = - torch.pi/2
    # q[:, 4] = torch.pi/4
    
    # # knee_range = torch.linspace(-torch.pi/10, torch.pi/10, 5)
    # knee_range = [torch.pi/10]
    # for knee_value in knee_range:
    #     # left foot
    #     q[:, 3] = - torch.pi/2 + knee_value
    #     q[:, 4] = - q[:, 2] - q[:, 3]
    #     biped.forward_kinematics_tree(q, 0)
    #     p01 = biped.T01_left[:, :3, 3] # hip yaw link
    #     p02 = biped.T02_left[:,:3, 3] # hip roll link
    #     p03 = biped.T03_left[:,:3, 3] # thigh pitch link
    #     p04 = biped.T04_left[:,:3, 3] # calkf pitch link
    #     p05 = biped.T05_left[:,:3, 3] # ankle pitch link
    #     p0e = biped.get_p0e(0) # foot sole link
    #     p0toe = biped.get_p0toe(0) # foot toe link
    #     p0heel = biped.get_p0heel(0) # foot heel link
        
    #     positions = torch.stack([p01[0], p02[0], p03[0], p04[0], p05[0], p0e[0]], dim=0)
    #     positions_ht = torch.stack([p0toe[0], p0heel[0]], dim=0)
    #     ax.plot(positions[:, 0].cpu(), positions[:, 1].cpu(), positions[:, 2].cpu(), marker='o', color='black')
    #     ax.plot(positions_ht[:, 0].cpu(), positions_ht[:, 1].cpu(), positions_ht[:, 2].cpu(), marker='o', color='black')
        
    #     # right foot
    #     q[:, 3] = - torch.pi/2 + knee_value
    #     biped.forward_kinematics_tree(q, 1)
    #     p01 = biped.T01_right[:, :3, 3]
    #     p02 = biped.T02_right[:,:3, 3]
    #     p03 = biped.T03_right[:,:3, 3]
    #     p04 = biped.T04_right[:,:3, 3]
    #     p05 = biped.T05_right[:,:3, 3]
    #     p0e = biped.get_p0e(1)
    #     p0toe = biped.get_p0toe(1)
    #     p0heel = biped.get_p0heel(1)
        
    #     positions = torch.stack([p01[0], p02[0], p03[0], p04[0], p05[0], p0e[0], p0toe[0]], dim=0)
    #     positions_ht = torch.stack([p0toe[0], p0heel[0]], dim=0)
    #     ax.plot(positions[:, 0].cpu(), positions[:, 1].cpu(), positions[:, 2].cpu(), marker='o', color='black')
    #     ax.plot(positions_ht[:, 0].cpu(), positions_ht[:, 1].cpu(), positions_ht[:, 2].cpu(), marker='o', color='black')
        
    # # plot torso
    # x_range = [-0.05, 0.05]
    # y_range = [-0.1, 0.1]
    # z_range = [-0.18, 0.18]

    # # Create the vertices of the rectangular prism.
    # vertices = np.array([[x, y, z] for x in x_range for y in y_range for z in z_range])
    # # Define the 6 faces of the prism.
    # faces = [
    #     [vertices[0], vertices[1], vertices[3], vertices[2]],  # Bottom face (z=0)
    #     [vertices[4], vertices[5], vertices[7], vertices[6]],  # Top face (z=1)
    #     [vertices[0], vertices[1], vertices[5], vertices[4]],  # Front face (y=0)
    #     [vertices[2], vertices[3], vertices[7], vertices[6]],  # Back face (y=3)
    #     [vertices[1], vertices[3], vertices[7], vertices[5]],  # Right face (x=2)
    #     [vertices[0], vertices[2], vertices[6], vertices[4]]   # Left face (x=0)
    # ]
    
    # box = Poly3DCollection(faces, facecolors='black', linewidths=1, edgecolors='black', alpha=0.25)
    # ax.add_collection3d(box)
    # ax.plot([0], [0], [0], marker='o', color='r')
    
    # plt.axis('equal')
    # plt.show()