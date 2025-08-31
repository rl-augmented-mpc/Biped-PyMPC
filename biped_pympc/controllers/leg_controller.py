from dataclasses import dataclass
from dataclasses import field
from typing import Union, Literal
import torch

from biped_pympc.core.robot.robot_factory import RobotFactory
from biped_pympc.core.data.robot_data import LegControllerData, LegControllerCommand

class LegController:
    def __init__(self, 
                 num_envs:int, 
                 num_leg:int, 
                 device:Union[torch.device, str]=torch.device("cpu"), 
                 robot: Literal["HECTOR", "T1"]="HECTOR"):
        self.num_envs = num_envs
        self.num_leg = num_leg
        self.device = device

        self.biped = RobotFactory(robot)(num_envs, device)
        self.pd_conf = self.biped.pd_conf
        self.num_dof = self.biped.num_dof
        
        self.data = LegControllerData(num_leg=num_leg, num_dof=self.num_dof, batch_size=num_envs, device=device)
        self.command = LegControllerCommand(num_leg=num_leg, num_dof=self.num_dof, batch_size=num_envs, device=device)
    
    # === execute these ============================
    def update_gait_data(self, contact_phase:torch.Tensor, swing_phase:torch.Tensor)->None:
        """
        Update internal state of leg controller gait data
        contact_phase: contact phase for each foot torch.Tensor of shape (batch_size, num_leg)
        swing_phase: swing phase for each foot torch.Tensor of shape (batch_size, num_leg)
        """
        self.data.contact_phase = contact_phase
        self.data.contact_bool[:, 0] = (contact_phase[:, 0] != -1).float()
        self.data.contact_bool[:, 1] = (contact_phase[:, 1] != -1).float()
        self.data.swing_phase = swing_phase
        self.data.swing_bool[:, 0] = (swing_phase[:, 0] != -1).float()
        self.data.swing_bool[:, 1] = (swing_phase[:, 1] != -1).float()
    
    def update_data(self, q:torch.Tensor, qd:torch.Tensor, tau:torch.Tensor)->None:
        """
        Update internal state of leg controller data
        q: measured joint angle  torch.Tensor of shape (batch_size, num_dof*num_leg)
        qd: measured joint velocity  torch.Tensor of shape (batch_size, num_dof*num_leg)
        tau: measured joint torque  torch.Tensor of shape (batch_size, num_dof*num_leg
        """
        # update joint state
        self.data.q = q.view(-1, self.data.num_leg, self.num_dof)
        self.data.qd = qd.view(-1, self.data.num_leg, self.num_dof)
        self.data.tau = tau.view(-1, self.data.num_leg, self.num_dof)
        
        # new kinematics
        self.biped.forward_kinematics(self.data.q[:, 0, :], 0)
        self.biped.forward_kinematics(self.data.q[:, 1, :], 1)
        self.data.p[:, 0, :] = self.biped.get_p0e(0)
        self.data.p[:, 1, :] = self.biped.get_p0e(1)
        
        # compute contact jacobian
        JL = self.biped.contact_jacobian(0)
        JR = self.biped.contact_jacobian(1)
        self.data.J[:, 0, :, :] = JL
        self.data.J[:, 1, :, :] = JR
        self.data.Jv[:, 0, :, :] = JL[:, :3, :]
        self.data.Jv[:, 1, :, :] = JR[:, :3, :]
        
        # update foot velocity
        vL = (JL[:, :3, :] @ self.data.qd[:, 0, :, None]).squeeze(-1)  # (batch_size, 3, 1)
        vR = (JR[:, :3, :] @ self.data.qd[:, 1, :, None]).squeeze(-1)  # (batch_size, 3, 1)
        self.data.v[:, 0, :] = vL
        self.data.v[:, 1, :] = vR
        
    def update_command(self)->None:
        # feedforward force and contact/stance phase/bool come from qp controller
        # pDes, vDes come from swing leg controller
        self.set_pd_gain()
        self.update_ff_torque()
        self.update_fb_torque()
        
    # === execute these ============================
    
    def set_pd_gain(self):
        for i in range(len(self.pd_conf.kp)):
            self.command.kpjoint[:, :, i] = self.pd_conf.kp[i]
        for i in range(len(self.pd_conf.kd)):
            self.command.kdjoint[:, :, i] = self.pd_conf.kd[i]
    
    def update_ff_torque(self):
        # Inverse dynamics (tau = J^T*f)
        stance_tau_left = (self.data.J[:, 0, :, :].transpose(1,2) @ self.command.feedfowardforce[:, 0, :].unsqueeze(-1)).squeeze(-1)
        stance_tau_right = (self.data.J[:, 1, :, :].transpose(1,2) @ self.command.feedfowardforce[:, 1, :].unsqueeze(-1)).squeeze(-1)
        zero_tau = torch.zeros_like(stance_tau_left)
        
        # Disable feedforward torque for swing leg
        self.command.tau[:, 0, :] = torch.where(self.data.contact_bool[:, 0].unsqueeze(-1).bool(), stance_tau_left, zero_tau)
        self.command.tau[:, 1, :] = torch.where(self.data.contact_bool[:, 1].unsqueeze(-1).bool(), stance_tau_right, zero_tau)
    
    def update_fb_torque(self):
        swing_qDes_left = self.biped.analytical_IK(self.command.pDes[:, 0, :], 0)
        swing_qDes_right = self.biped.analytical_IK(self.command.pDes[:, 1, :], 1)
        stance_qDes_left = torch.zeros_like(swing_qDes_left)
        stance_qDes_right = torch.zeros_like(swing_qDes_right)
        
        self.command.qDes[:, 0, :] = torch.where(self.data.contact_bool[:, 0].unsqueeze(-1).bool(), stance_qDes_left, swing_qDes_left)
        self.command.qDes[:, 1, :] = torch.where(self.data.contact_bool[:, 1].unsqueeze(-1).bool(), stance_qDes_right, swing_qDes_right)
        
        swing_qdDes_left = torch.bmm(self.data.Jv[:, 0, :, :].transpose(1, 2), self.command.vDes[:, 0, :].unsqueeze(-1)).squeeze(-1)
        swing_qdDes_right = torch.bmm(self.data.Jv[:, 1, :, :].transpose(1, 2), self.command.vDes[:, 1, :].unsqueeze(-1)).squeeze(-1)
        stance_qdDes_left = torch.zeros_like(swing_qdDes_left)
        stance_qdDes_right = torch.zeros_like(swing_qdDes_right)
        
        self.command.qdDes[:, 0, :] = torch.where(self.data.contact_bool[:, 0].unsqueeze(-1).bool(), stance_qdDes_left, swing_qdDes_left)
        self.command.qdDes[:, 1, :] = torch.where(self.data.contact_bool[:, 1].unsqueeze(-1).bool(), stance_qdDes_right, swing_qdDes_right)
        
        self.command.kpjoint[self.data.contact_bool[:, 0].nonzero(), 0, :] *= 0.0
        self.command.kpjoint[self.data.contact_bool[:, 1].nonzero(), 1, :] *= 0.0

    def reset(self, env_id:torch.Tensor)->None:
        self.data.zero(env_id)
        self.command.zero(env_id)

if __name__ == "__main__":
    num_envs = 4096
    num_leg = 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    leg_controller = LegController(num_envs, num_leg, device)
    
    q = torch.zeros(num_envs, num_leg*5, device=device)
    qd = torch.zeros(num_envs, num_leg*5, device=device)
    tau = torch.zeros(num_envs, num_leg*5, device=device)
    leg_controller.update_data(q, qd, tau)
    leg_controller.update_command()