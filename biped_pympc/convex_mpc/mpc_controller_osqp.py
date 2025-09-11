import os
from time import time
from typing import Union, Tuple
import numpy as np
import casadi
import torch
import osqp

from biped_pympc.configuration.configuration import MPCConf
from biped_pympc.cusadi import CASADI_FUNCTION_DIR
from biped_pympc.convex_mpc.base_controller import BaseMPCController

class MPCControllerOSQP(BaseMPCController):
    def __init__(self, num_envs:int, device:Union[torch.device, str], num_legs:int, cfg:MPCConf):
        super().__init__(num_envs, device, num_legs, cfg)
        self.init_qp = True
    
    """
    initialization.
    """
    def init_solver(self):
        # load matrix fomer
        qp_former = casadi.Function.load(
            os.path.join(CASADI_FUNCTION_DIR, "srbd_qp_mat.casadi")
            )
        self.qp_former = qp_former  # Use regular CasADi function, not CuSaDi
        
        # setup osqp solver
        self.qp_solver = osqp.OSQP()
    
    """
    operations.
    """

    
    def run(self)->Tuple[torch.Tensor, torch.Tensor]:
        t0 = time()
        self.compute_knot_points()
        self.compute_horizon_state()
        self.set_initial_state()
        self.compute_reference_trajectory()
        
        # form QP matrices
        x0 = self.x0[0, :12].cpu().numpy()
        x = torch.ones(12*self.horizon_length).cpu().numpy()
        u = torch.ones(12*self.horizon_length).cpu().numpy()
        x_ref = self.x_ref[0, :, :12].contiguous().view(-1).cpu().numpy()
        dt = self.dt_mpc[0].item()
        m = self.mass
        friction_coef = self.mu
        R_b = self.R_body[0, 0].cpu().numpy()
        I_w = (self.state_estimate_data.rotation_body @ 
            self.biped.I_body.unsqueeze(0) @
            self.state_estimate_data.rotation_body.transpose(1, 2))[0].cpu().numpy()
        body_pos = self.state_estimate_data.root_position[0, :].cpu().numpy()
        left_foot_pos = self.state_estimate_data.foot_position[0, 0, :].cpu().numpy()
        right_foot_pos = self.state_estimate_data.foot_position[0, 1, :].cpu().numpy()
        contact_table = self.contact_table[0, :, :].cpu().numpy()
        Q = self.Q.cpu().numpy()
        R = self.R.cpu().numpy()
        residual_lin_accel = self.residual_lin_accel[0].cpu().numpy()
        residual_ang_accel = self.residual_ang_accel[0].cpu().numpy()

        H, f, A_block, b_block, G_block, d_block = self.qp_former(
            x0, x, u, x_ref, dt, m, friction_coef, R_b, I_w, 
            body_pos, left_foot_pos, right_foot_pos, contact_table, 
            Q, R, 
            residual_lin_accel, residual_ang_accel
            )
        
        # form sparse matrices
        A = casadi.vertcat(A_block, G_block)
        lb = casadi.vertcat(b_block, -np.inf * np.ones((d_block.shape[0], 1)))
        ub = casadi.vertcat(b_block, d_block)
        H_sp = H.sparse()
        f_sp = f.full()
        A_sp = A.sparse()
        lb_sp = lb.full()
        ub_sp = ub.full()
        # print(f"qp mat calc time: {1000*(time() - t0):.4f} ms")s
        
        # setup qp
        t0 = time()
        if self.init_qp:
            self.qp_solver.setup(H_sp, f_sp, A_sp, lb_sp, ub_sp, rho=0.5, eps_abs=1e-12, verbose=False)
            self.init_qp = False
        else:
            self.qp_solver.update(Px=H_sp.data, Ax=A_sp.data, q=f_sp, l=lb_sp, u=ub_sp)
        # print(f"osqp setup time: {1000*(time() - t0):.4f} ms")
        
        # solve qp 
        t0 = time()
        res = self.qp_solver.solve()
        sol = torch.from_numpy(res.x).float().unsqueeze(0).to(self.device)
        cost = torch.tensor([res.info.obj_val], dtype=torch.float32, device=self.device)
        # print(f"qp solver time: {1000*(time() - t0):.4f} ms")
        u_control = sol[:, 12*self.horizon_length:12*self.horizon_length+12] # casadi qp former
        
        # solutions are in global coordinate
        t0 = time()
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
        # print(f"solution process time: {1000*(time() - t0):.4f} ms")

        return foot_wrench, cost