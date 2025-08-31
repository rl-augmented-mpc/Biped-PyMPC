import os
import sys
from time import time
from typing import Union, Tuple
from dataclasses import dataclass
import casadi
import numpy as np
import torch

from biped_pympc.configuration.configuration import MPCConf
from biped_pympc.convex_mpc.base_controller import BaseMPCController
from biped_pympc.cusadi import CASADI_FUNCTION_DIR

class MPCControllerCasadi(BaseMPCController):
    def __init__(self, num_envs:int, device:Union[torch.device, str], num_legs:int, cfg:MPCConf):
        super().__init__(num_envs, device, num_legs, cfg)
    
    """
    initialization.
    """
    
    def init_solver(self):
        # load matrix fomer
        qp_former = casadi.Function.load(
            os.path.join(CASADI_FUNCTION_DIR, "srbd_qp_mat.casadi")
            )
        self.qp_former = qp_former  # Use regular CasADi function, not CuSaDi
        
        # load function
        # solver_filename = os.path.join(CASADI_FUNCTION_DIR, "mpc_multiple_iter_5_solver_240v_140eq_160ineq.casadi")
        solver_filename = os.path.join(CASADI_FUNCTION_DIR, "mpc_multiple_iter_20_solver_240v_140eq_160ineq.casadi")
        self.qp_solver = casadi.Function.load(solver_filename)
        print(f"Solver loaded successfully. Number of inputs: {self.qp_solver.n_in()}, Number of outputs: {self.qp_solver.n_out()}")
    
    """
    operations.
    """

    
    def run(self)->Tuple[torch.Tensor, torch.Tensor]:
        self.compute_knot_points()
        self.compute_horizon_state()
        self.set_initial_state()
        self.compute_reference_trajectory()
        
        # # form qp matrices manually in pytorch
        # H, f, A_block, b_block, G_block, d_block = create_srbd_qp_matrices(
        #     self.R_body.view(-1, self.horizon_length*3*3),
        #     self.I_world_inv.view(-1, self.horizon_length*3*3),
        #     self.left_foot_pos_skew.view(-1, self.horizon_length*3*3),
        #     self.right_foot_pos_skew.view(-1, self.horizon_length*3*3),
        #     self.contact_table.view(-1, 2*self.horizon_length),
        #     self.x_ref.view(-1, self.horizon_length*13), # <- add 1 at the end of last dim to make (batch, horizon, 13)
        #     self.Q, self.R, self.x0, self.mass, self.mu, self.dt, self.iteration_between_mpc
        # )
        # # end of pytorch qp former
        
        ### 1. Create QP matrices using casadi ###
        # Pick first batch and send it back to CPU 
        # This is why casadi controller does not support batch processing
        x0 = self.x0[0, :12].cpu().numpy()
        x = torch.ones(12*self.horizon_length, device=self.device).cpu().numpy()
        u = torch.ones(12*self.horizon_length, device=self.device).cpu().numpy()
        x_ref = self.x_ref[0, :, :12].contiguous().view(-1).cpu().numpy()
        R_b = self.R_body[0, 0].cpu().numpy()
        I_w = (self.state_estimate_data.rotation_body @ 
            self.biped.I_body.unsqueeze(0) @
            self.state_estimate_data.rotation_body.transpose(1, 2))[0].cpu().numpy()
        body_pos = self.state_estimate_data.root_position[0, :].cpu().numpy()
        left_foot_pos = self.state_estimate_data.foot_position[0, 0, :].cpu().numpy()
        right_foot_pos = self.state_estimate_data.foot_position[0, 1, :].cpu().numpy()
        contact_table = self.contact_table[0, :, :].cpu().numpy()
        
        dt_mpc = self.dt_mpc[0].item()
        m = self.mass
        friction_coef = self.mu
        Q = self.Q.cpu().numpy()
        R = self.R.cpu().numpy()
        
        residual_lin_accel = self.residual_lin_accel[0].cpu().numpy()
        residual_ang_accel = self.residual_ang_accel[0].cpu().numpy()
        
        # Call CasADi function directly (same as test file)
        H, f, A, b, G, d = self.qp_former(
            x0, x, u, x_ref, dt_mpc, m, friction_coef,
            R_b, I_w, 
            body_pos, left_foot_pos, right_foot_pos, 
            contact_table,
            Q, R, 
            residual_lin_accel, residual_ang_accel
        )
        
        # Convert CasADi objects to numpy arrays and reshape
        H_np = np.array(H.full())
        G_np = np.array(G.full())
        A_np = np.array(A.full())
        f_np = np.array(f.full()).flatten()
        d_np = np.array(d.full()).flatten()
        b_np = np.array(b.full()).flatten()
        
        # Get sparse matrix nonzeros
        H_nnz = H.nonzeros()
        G_nnz = G.nonzeros()
        A_nnz = A.nonzeros()
        
        # Get dimensions
        nz = H.shape[0]  # number of variables
        num_eq = A.shape[0]  # number of equality constraints
        num_ineq = G.shape[0]  # number of inequality constraints
        
        # Convert to column vectors for solver
        H_nnz_col = np.array(H_nnz).reshape(-1, 1)
        G_nnz_col = np.array(G_nnz).reshape(-1, 1)
        A_nnz_col = np.array(A_nnz).reshape(-1, 1)
        f_col = f_np.reshape(-1, 1)
        d_col = d_np.reshape(-1, 1)
        b_col = b_np.reshape(-1, 1)
        
        # Use simple initial guess like in multiple_iter.py
        z_col = np.concatenate([x, u]).reshape(-1, 1)  # x_init should be full solution vector
        
        # Call single iteration CasADi solver with external iteration loop
        solve_start = time()
        
        # Initialize PDIPM variables
        x_current, s_current, z_current, y_current = self.initialize_pdipm_variables(
            nz, num_eq, num_ineq, G_np, d_np
        )
        
        # External iteration loop
        MAX_ITER = 1
        for iter_num in range(MAX_ITER):
            # === TRUST THE SINGLE ITERATION SOLVER (EXACT COPY OF MULTIPLE ITERATION LOGIC) ===
            # Call single iteration solver
            x_new, s_new, z_new, y_new, residuals, mu = self.qp_solver(
                H_nnz_col, G_nnz_col, A_nnz_col, f_col, d_col, b_col,
                x_current, s_current, z_current, y_current
            )
            
            # Convert CasADi outputs to numpy
            x_new_np = np.array(x_new).flatten()
            s_new_np = np.array(s_new).flatten()
            z_new_np = np.array(z_new).flatten()
            y_new_np = np.array(y_new).flatten()
            
            
            # Always use the solver output (trust it completely)
            x_current = x_new_np
            s_current = s_new_np
            z_current = z_new_np
            y_current = y_new_np
        
        # Get the final solution
        x_opt_np = x_current
        sol = torch.from_numpy(x_opt_np).to(self.device).to(torch.float32)

        cost = torch.tensor([0.0], dtype=torch.float32, device=self.device)
        u_control = sol[12*self.horizon_length:12*self.horizon_length+12] # casadi qp former - 1D tensor
        
        # solutions are in global coordinate
        left_grf = u_control[:3]
        right_grf = u_control[3:6]
        left_grm = u_control[6:9]
        right_grm = u_control[9:12]
        left_grm[0] = 0.0
        right_grm[0] = 0.0
        
        # transform wrench to body frame
        left_grf_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ left_grf.unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
        left_grm_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ left_grm.unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
        right_grf_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ right_grf.unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
        right_grm_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ right_grm.unsqueeze(-1)).squeeze(-1) # (batch_size, 3)
        
        foot_wrench = torch.cat([-left_grf_body, -left_grm_body, -right_grf_body, -right_grm_body], dim=1).reshape(-1, 2, 6) # (batch_size, 2, 6)
        
        return foot_wrench, cost
    
    """
    internal methods
    """

    def initialize_pdipm_variables(self, nz: int, num_eq: int, num_ineq: int, G_np: np.ndarray, d_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize primal-dual interior point method variables with proper initialization
        """
        # Initial solution - use simple zero initialization
        x_current = np.zeros(nz)
        
        # Initialize slack variables s > 0 - match multiple-iteration solver logic
        Gx = G_np @ x_current
        s_current = np.maximum(d_np - Gx, 1.0)  # s = max(h - Gx, 1.0) like multiple-iteration solver
        
        # Initialize dual variables z > 0 - match multiple-iteration solver logic
        z_current = np.ones(num_ineq)  # z = ones(num_ineq) like multiple-iteration solver
        
        # Initialize equality constraint dual variables y
        y_current = np.zeros(num_eq)
        
        return x_current, s_current, z_current, y_current