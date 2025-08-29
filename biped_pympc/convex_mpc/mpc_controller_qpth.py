import os
from time import time
from typing import Union, Tuple
from dataclasses import dataclass
import casadi
import torch
from qpth.qp import QPFunction, QPSolvers

from biped_pympc.configuration.configuration import MPCConf
from biped_pympc.cusadi import CASADI_FUNCTION_DIR, CusadiFunction
from biped_pympc.convex_mpc.base_controller import BaseMPCController

class MPCControllerQPTh(BaseMPCController):
    def __init__(self, num_envs:int, device:Union[torch.device, str], num_legs:int, cfg:MPCConf):
        super().__init__(num_envs, device, num_legs, cfg)
    
    """
    initialization.
    """
    
    def init_solver(self):
        # qp solver
        self.qp_solver = QPFunction(verbose=-1,
                  check_Q_spd=False,
                  eps=1e-3,
                  solver=QPSolvers.PDIPM_BATCHED)
        
        qp_former_filename = os.path.join(CASADI_FUNCTION_DIR, "srbd_qp_mat.casadi")
        qp_former = casadi.Function.load(qp_former_filename)
        self.qp_former = CusadiFunction(qp_former, num_instances=self.num_envs)
    
    
    """
    operations.
    """
    
    def run(self)->Tuple[torch.Tensor, torch.Tensor]:
        self.compute_knot_points()
        self.compute_horizon_state()
        self.set_initial_state()
        self.compute_reference_trajectory()
        
        ## DEPRECATED!!!
        # # form qp matrices manually in pytorch
        # H, f, A_block, b_block, G_block, d_block = create_srbd_qp_matrices(
        #     self.R_body.view(-1, self.horizon_length*3*3),
        #     self.I_world_inv.view(-1, self.horizon_length*3*3),
        #     self.left_foot_pos_skew.view(-1, self.horizon_length*3*3),
        #     self.right_foot_pos_skew.view(-1, self.horizon_length*3*3),
        #     self.contact_table.view(-1, 2*self.horizon_length),
        #     self.x_ref.view(-1, self.horizon_length*13),
        #     self.Q, self.R, self.x0, self.mass, self.mu, self.dt_mpc
        # )
        # # end of pytorch qp former
        

        # CusaDi QP former
        # some useful constants
        nx = 12
        nu = 12

        ### 1. Create QP matrices using cusadi ###
        x0 = self.x0.contiguous()  # (num_envs, 12) - already 12-state
        x = torch.ones(self.num_envs, 12*self.horizon_length, device=self.device).contiguous()  # (num_envs, 12*horizon)
        u = torch.ones(self.num_envs, 12*self.horizon_length, device=self.device).contiguous()  # (num_envs, 12*horizon)
        x_ref = self.x_ref.contiguous().view(self.num_envs, -1)  # (num_envs, 12*horizon) - already 12-state
        R_b = self.R_body[:, 0, :, :].contiguous().view(self.num_envs, -1)  # (num_envs, 9)
        I_w = (self.state_estimate_data.rotation_body @ 
            self.biped.I_body.unsqueeze(0) @
            self.state_estimate_data.rotation_body.transpose(1, 2)).contiguous().view(self.num_envs, -1)  # (num_envs, 9)
        body_pos = self.state_estimate_data.root_position.contiguous()  # (num_envs, 3)
        left_foot_pos = self.state_estimate_data.foot_position[:, 0, :].contiguous()  # (num_envs, 3)
        right_foot_pos = self.state_estimate_data.foot_position[:, 1, :].contiguous()  # (num_envs, 3)
        contact_table = self.contact_table.contiguous().view(self.num_envs, -1)  # (num_envs, horizon*2)
    
        dt_mpc = self.dt_mpc.unsqueeze(1).double()
        m = self.mass * torch.ones(self.num_envs, 1, device=self.device, dtype=torch.double)
        friction_coef = self.mu * torch.ones(self.num_envs, 1, device=self.device, dtype=torch.double)
        Q = self.Q.double().unsqueeze(0).repeat(self.num_envs, 1)
        R = self.R.double().unsqueeze(0).repeat(self.num_envs, 1)

        residual_lin_accel = self.residual_lin_accel.double()
        residual_ang_accel = self.residual_ang_accel.double()
        
        # Prepare inputs for CuSaDi function with proper tensor formatting
        inputs = [
            x0.double(),  # (num_envs, 12)
            x.double(),   # (num_envs, 12*horizon)
            u.double(),   # (num_envs, 12*horizon)
            x_ref.double(),  # (num_envs, 12*horizon)
            dt_mpc, # (num_envs, 1)
            m, # (num_envs, 1)
            friction_coef, # (num_envs, 1)
            R_b.double(),  # (num_envs, 3, 3)
            I_w.double(),  # (num_envs, 3, 3)
            body_pos.double(),  # (num_envs, 3)
            left_foot_pos.double(),  # (num_envs, 3)
            right_foot_pos.double(),  # (num_envs, 3)
            contact_table.double(),  # (num_envs, horizon, 2)
            Q,  # (num_envs, 12)
            R,   # (num_envs, 12)
            residual_lin_accel,  # (num_envs, 3)
            residual_ang_accel   # (num_envs, 3)
        ]
        
        # Evaluate CuSaDi function (populate internal outputs)
        qp_gen_start = time()
        self.qp_former.evaluate(inputs)
        qp_gen_time = time() - qp_gen_start
        
        # Retrieve dense outputs from internal buffers
        H = self.qp_former.getDenseOutput(0)  # (B, nz, nz)
        f = self.qp_former.getDenseOutput(1).squeeze(-1)  # (B, nz)
        A = self.qp_former.getDenseOutput(2)  # (B, num_eq, nz)
        b = self.qp_former.getDenseOutput(3).squeeze(-1)  # (B, num_eq)
        G = self.qp_former.getDenseOutput(4)  # (B, num_ineq, nz)
        d = self.qp_former.getDenseOutput(5).squeeze(-1)  # (B, num_ineq)
        
        # solve qp 
        # t0 = time()
        sol = self.qp_solver(
            H.double(), 
            f.double(), 
            G.double(), 
            d.double(), 
            A.double(), 
            b.double()
            ).float()
        # print(f"qp solver time: {1000*(time() - t0):.4f} ms")
        cost = (0.5 * sol[:, :, None].transpose(1, 2) @ H.float() @ sol[:, :, None] + f[:, :, None].transpose(1, 2).float() @ sol[:, :, None]).squeeze(-1)
        u_control = sol[:, nx*self.horizon_length:nx*self.horizon_length+nu]  # (num_envs, 12)
        
        # solutions are in global coordinate
        left_grf = u_control[:, :3]
        right_grf = u_control[:, 3:6]
        left_grm = u_control[:, 6:9]
        right_grm = u_control[:, 9:12]
        left_grm[:, 0] = 0.0
        right_grm[:, 0] = 0.0
        
        # transform wrench to body frame
        left_grf_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ left_grf.float().unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
        left_grm_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ left_grm.float().unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
        right_grf_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ right_grf.float().unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
        right_grm_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ right_grm.float().unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
        foot_wrench = torch.cat([-left_grf_body, -left_grm_body, -right_grf_body, -right_grm_body], dim=1).reshape(-1, 2, 6) # (batch_size, 2, 6)
        
        return foot_wrench, cost