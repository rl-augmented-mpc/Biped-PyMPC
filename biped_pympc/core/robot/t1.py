from dataclasses import dataclass, field
import os
from pathlib import Path
import casadi as cs
import torch

from biped_pympc.core.robot.biped import Biped
from biped_pympc.utils.math_utils import rot_x, rot_z
from biped_pympc.cusadi import CASADI_FUNCTION_DIR


class T1FK:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = cache_dir

        self.left_foot_fk = self._load_function("t1_fk_left")
        self.right_foot_fk = self._load_function("t1_fk_right")
        self.left_foot_jacobian = self._load_function("t1_jac_left")
        self.right_foot_jacobian = self._load_function("t1_jac_right")

    def _load_function(self, name: str) -> cs.Function:
        path = os.path.join(self.cache_dir, f"{name}.casadi")
        if not os.path.exists(path):
            print(self.cache_dir)
            raise FileNotFoundError(f"CasADi function file '{name}.casadi' not found in {self.cache_dir}. "
                                    f"Run T1FunctionGenerator first.")
        return cs.Function.load(path)

    def compute_fk(self, q, foot: str = "left") -> cs.DM:
        if foot == "left":
            return self.left_foot_fk(q)
        elif foot == "right":
            return self.right_foot_fk(q)
        else:
            raise ValueError("Invalid foot name. Use 'left' or 'right'.")

    def compute_jacobian(self, q, foot: str = "left") -> cs.DM:
        if foot == "left":
            return self.left_foot_jacobian(q)
        elif foot == "right":
            return self.right_foot_jacobian(q)
        else:
            raise ValueError("Invalid foot name. Use 'left' or 'right'.")
        
@dataclass
class DofCfg:
    kp: list[float] = field(default_factory=lambda: [20.0, 20.0, 20.0, 20.0, 15.0, 15.0])
    kd: list[float] = field(default_factory=lambda: [1.0, 1.0, 0.7, 0.7, 0.7, 0.7])
    torque_limit: list[float] = field(default_factory=lambda: [33.5, 33.5, 33.5, 67.0, 33.5, 33.5, 33.5, 33.5, 33.5, 67.0, 33.5, 33.5])

class T1(Biped):
    def __init__(self, num_envs:int, device:torch.device):
        super().__init__(num_envs, device)
        
        # left leg hip yaw configuration
        self.hip_yaw = torch.tensor([-0.00, 0.047, -0.126], device=self.device)
        self.hip_roll = torch.tensor([0.0465, 0.015, -0.0705], device=self.device)
        
        # pd conf
        self.pd_conf = DofCfg()
        
        self.define_dynamics_parameter()
        self.initialize_kinematics()

        # Internal caches
        # self._last_joint_angle = None
        self._cached_joint_angle = {0: None, 1: None}
        self._cached_foot_pos = {0: None, 1: None}
        
    def define_dynamics_parameter(self):
        self.mass = 40.0
        self.I_body = torch.tensor([[0.5413, 0.0, 0.0],
                                    [0.0, 0.5200, 0.0],
                                    [0.0, 0.0, 0.0691]], device=self.device)
        self.mu = 1.0
        
    @property
    def num_dof(self):
        return 6
    
    def hip_horizontal_location(self, leg:int)->torch.Tensor:
        """
        Get center of gravity when one leg is in swing phase. 
        It is usually computed as projection of hip position
        """
        side = 1*(1-leg) -1*leg # leg0: side=1, leg1: side=-1
        cog_location = torch.zeros(self.num_envs, 3, device=self.device)
        cog_location[:, 0] = 0.0625 - 0.014
        cog_location[:, 1] = side * 0.106
        return cog_location
    
    def initialize_kinematics(self)->None:
        # cache_dir = str(Path(__file__).parent.parent.parent / "casadi/function")
        self.fk_model = T1FK(cache_dir=CASADI_FUNCTION_DIR)
    
    def forward_kinematics(self, joint_angle: torch.Tensor, leg:int)->None:
        assert joint_angle.shape[-1] == self.num_dof

        self._cached_joint_angle[leg] = joint_angle.clone()
        
        # TODO: cpu specific, later change to batch process
        # pick first batch
        q_np = joint_angle[0, :].cpu().numpy()

        if leg == 0:
            pos = self.fk_model.left_foot_fk(q_np).full().squeeze()
        else:
            pos = self.fk_model.right_foot_fk(q_np).full().squeeze()

        self._cached_foot_pos[leg] = torch.from_numpy(pos).unsqueeze(0).to(self.device, dtype=torch.float32)
    
    def get_p0e(self, leg:int)->torch.Tensor:
        """
        retrieve foot end effector position wrt base frame
        """
        if self._cached_foot_pos[leg] is None:
            raise RuntimeError("Call forward_kinematics first")
        return self._cached_foot_pos[leg]
    
    def contact_jacobian(self, leg:int)->torch.Tensor:
        """
        get contact Jacobian given current forward kinematics results.
        
        Args:
            leg (int): Leg identifier (0 for left, 1 for right).
        """
        if self._cached_joint_angle[leg] is None:
            raise RuntimeError("Call forward_kinematics first")

        # TODO: cpu specific, later change to batch process
        # pick first batch
        q_np = self._cached_joint_angle[leg][0, :].cpu().numpy()

        if leg == 0:
            J = self.fk_model.left_foot_jacobian(q_np).full()
        else:
            J = self.fk_model.right_foot_jacobian(q_np).full()
        return torch.from_numpy(J).unsqueeze(0).to(self.device, dtype=torch.float32)
    
    def analytical_IK(self, p_foot_des_b: torch.Tensor, leg:int) -> torch.Tensor:
        """
        Compute the inverse kinematics for a bipedal robot leg with batch support.
        Since we only track 3D position, there can be only 3 variables to solve for.
        This is why we fix hip yaw angle and ankle roll angle to be 0 and ankle pitch angle to be aligned with the torso pitch.
        
        Args:
            p_foot_des_b (torch.Tensor): Desired foot position in hip frame (batch_size x 3).
            leg (int): Leg identifier (0 for left, 1 for right).

        Returns:
            torch.Tensor: Joint angles (batch_size, num_dof)
        """

        dtype, device = p_foot_des_b.dtype, p_foot_des_b.device
        side = 1 * (1-leg) - 1 * leg  # 1: for left leg, -1: for right leg

        # constants (meters)
        r_torso_to_waist = [0.0625, 0, -0.1155]
        r_waist_to_hip_pitch = [0, 0.106, 0]
        r_hip_pitch_to_hip_roll = [0, 0, -0.02]
        r_hip_roll_to_hip_yaw = [0, 0, -0.081854]
        r_hip_yaw_to_knee = [-0.014, 0, -0.134]
        r_knee_to_ankle_pitch = [0, 0, -0.28]
        r_ankle_pitch_to_ankle_roll = [0, 0.00025, -0.012]
        r_ankle_roll_to_foot = [0, 0, -0.035192] # TODO: double check this value
        
        r_torso_to_hip = torch.zeros_like(p_foot_des_b, dtype=dtype, device=device)
        r_torso_to_hip[:, 0] = r_torso_to_waist[0] + r_waist_to_hip_pitch[0]
        r_torso_to_hip[:, 1] = side * (r_torso_to_waist[1] + r_waist_to_hip_pitch[1])
        r_torso_to_hip[:, 2] = r_torso_to_waist[2] + r_waist_to_hip_pitch[2]
        
        r_ankle_roll_to_ee = torch.zeros_like(p_foot_des_b, dtype=dtype, device=device)
        r_ankle_roll_to_ee[:, 0] = r_ankle_pitch_to_ankle_roll[0] + r_ankle_roll_to_foot[0]
        r_ankle_roll_to_ee[:, 1] = side * (r_ankle_pitch_to_ankle_roll[1] + r_ankle_roll_to_foot[1])
        r_ankle_roll_to_ee[:, 2] = r_ankle_roll_to_foot[2]  # z
        
        # thigh length and shank length
        L1 = -(r_hip_pitch_to_hip_roll[2] + r_hip_roll_to_hip_yaw[2] + r_hip_yaw_to_knee[2])  # hip->knee
        L2 = -(r_knee_to_ankle_pitch[2] + r_ankle_pitch_to_ankle_roll[2])  # knee->ankle-roll
        knee_x_offset = r_hip_yaw_to_knee[0]

        # hip -> ankle-roll vector
        v = p_foot_des_b - r_torso_to_hip - r_ankle_roll_to_ee

        # hip roll
        hip_roll = torch.atan2(v[:, 1], -v[:, 2])

        # rotate by -hip_roll about x to get sagittal components
        hip_roll_cos, hip_roll_sin = torch.cos(hip_roll), torch.sin(hip_roll)
        xs = v[:, 0] - knee_x_offset
        zs = -v[:, 1] * hip_roll_sin + v[:, 2] * hip_roll_cos

        # planar 2R IK in x-z plane
        d = torch.sqrt(xs*xs + zs*zs)

        cos_beta = ((L1*L1 + d*d - L2*L2) / (2*L1*d + 1e-6)).clamp(-1.0, 1.0)
        beta = torch.acos(cos_beta)

        cos_k = ((L1*L1 + L2*L2 - d*d) / (2*L1*L2 + 1e-6)).clamp(-1.0, 1.0)
        knee_pitch = torch.pi - torch.acos(cos_k)

        alpha = torch.atan2(xs, -zs)
        hip_pitch = alpha - beta

        ankle_pitch = -(hip_pitch + knee_pitch)
        
        num_dof = self.num_dof
        q = torch.zeros(p_foot_des_b.shape[0], num_dof, device=p_foot_des_b.device)  # solution joint angles
        q[:, 0] = hip_pitch
        q[:, 1] = hip_roll
        q[:, 2] = 0.0  # hip yaw is fixed to 0
        q[:, 3] = knee_pitch
        q[:, 4] = ankle_pitch
        q[:, 5] = 0.0  # ankle roll is fixed to
        return q
        
        

if __name__ == "__main__":
    import torch

    # Setup
    device = torch.device("cpu")
    num_envs = 1
    robot = T1(num_envs=num_envs, device=device)

    q = torch.tensor([
        0.0,   0.0,  0.0,   # Left_Hip_Pitch, Left_Hip_Roll, Left_Hip_Yaw
        0.0,   0.0,  0.0,   # Left_Knee_Pitch, Left_Ankle_Pitch, Left_Ankle_Roll
        0.0,   0.0,  0.0,   # Right_Hip_Pitch, Right_Hip_Roll, Right_Hip_Yaw
        0.0,   0.0,  0.0    # Right_Knee_Pitch, Right_Ankle_Pitch, Right_Ankle_Roll
    ], device=device)
    q = q.unsqueeze(0).repeat(num_envs, 1)  # Shape: (num_envs, 12)

    q_left = q[:, :6]
    q_right = q[:, 6:]

    # Test Left Leg FK
    print("=== Left Leg ===")
    robot.forward_kinematics(q_left, leg=0)
    pos_left = robot.get_p0e(leg=0)
    jac_left = robot.contact_jacobian(leg=0)
    print("Foot Position (Left) shape:", pos_left.shape)
    print("Foot Position (Left):", pos_left)
    print("Jacobian (Left) shape:", jac_left.shape)
    print("Jacobian (Left):\n", jac_left)

    # Test Right Leg FK
    print("\n=== Right Leg ===")
    robot.forward_kinematics(q_right, leg=1)
    pos_right = robot.get_p0e(leg=1)
    jac_right = robot.contact_jacobian(leg=1)
    print("Foot Position (Right):", pos_right)
    print("Jacobian (Right):\n", jac_right)