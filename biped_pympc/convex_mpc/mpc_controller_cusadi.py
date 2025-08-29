import os
import sys
from time import time
from typing import Union, Tuple
from dataclasses import dataclass
import numpy as np
import casadi
import torch

from biped_pympc.configuration.configuration import MPCConf
from biped_pympc.cusadi import CASADI_FUNCTION_DIR, CusadiFunction
from biped_pympc.convex_mpc.base_controller import BaseMPCController


class MPCControllerCusadi(BaseMPCController):
    def __init__(self, num_envs:int, device:Union[torch.device, str], num_legs:int, cfg:MPCConf):
        super().__init__(num_envs, device, num_legs, cfg)
    
    """
    initialization.
    """
    
    def init_solver(self):
        """
        Load cusadi functions to evaluate.
        """
        # solver_filename = os.path.join(CASADI_FUNCTION_DIR, "mpc_multiple_iter_solver_240v_140eq_160ineq.casadi")
        solver_filename = os.path.join(CASADI_FUNCTION_DIR, "mpc_multiple_iter_5_solver_240v_140eq_160ineq.casadi")
        qp_solver = casadi.Function.load(solver_filename)
        self.qp_solver = CusadiFunction(qp_solver, num_instances=self.num_envs)
        
        qp_former_filename = os.path.join(CASADI_FUNCTION_DIR, "srbd_qp_mat.casadi")
        qp_former = casadi.Function.load(qp_former_filename)
        self.qp_former = CusadiFunction(qp_former, num_instances=self.num_envs)
        
        return self.qp_solver
    
    """
    operations.
    """

    
    def run(self)->Tuple[torch.Tensor, torch.Tensor]:
        self.compute_knot_points()
        self.compute_horizon_state()
        self.set_initial_state()
        self.compute_reference_trajectory()
        
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

        # Retrieve sparse outputs from internal buffers
        H_sparse = self.qp_former.outputs_sparse[0]  # (B, num_nonzeros) - sparse
        f_sparse = self.qp_former.outputs_sparse[1]  # (B, nz) - sparse
        A_sparse = self.qp_former.outputs_sparse[2]  # (B, num_nonzeros) - sparse
        b_sparse = self.qp_former.outputs_sparse[3]  # (B, num_eq) - sparse
        G_sparse = self.qp_former.outputs_sparse[4]  # (B, num_nonzeros) - sparse
        d_sparse = self.qp_former.outputs_sparse[5]  # (B, num_ineq) - sparse

        # Convert batched sparse data to tensors for solver 
        H_col = H_sparse.double().contiguous()  # (self.num_envs, num_nonzeros)
        G_col = G_sparse.double().contiguous()  # (self.num_envs, num_nonzeros)
        A_col = A_sparse.double().contiguous()  # (self.num_envs, num_nonzeros)
        f_col = f_sparse.reshape(self.num_envs, -1, 1).double().contiguous()  # (self.num_envs, nz, 1)
        d_col = d_sparse.reshape(self.num_envs, -1, 1).double().contiguous()  # (self.num_envs, num_ineq, 1)
        b_col = b_sparse.reshape(self.num_envs, -1, 1).double().contiguous()  # (self.num_envs, num_eq, 1)
        
       
        # Get dimensions from the first environment
        nz = H.shape[1]  # number of variables
        num_eq = A.shape[1]  # number of equality constraints
        num_ineq = G.shape[1]  # number of inequality constraints


        #### 2. Solve QP using cusadi pdpipm solver ####
        
        solve_start = time()

        # Initialize PDIPM variables for all environments (as PyTorch tensors)
        x_current = torch.zeros(self.num_envs, nz, device=self.device, dtype=torch.float64)
        s_current = torch.maximum(d - (G @ x_current.unsqueeze(2)).squeeze(2), torch.ones_like(d))
        z_current = torch.ones(self.num_envs, num_ineq, device=self.device, dtype=torch.float64)
        y_current = torch.ones(self.num_envs, num_eq, device=self.device, dtype=torch.float64)
        
        # Solve newton steps
        MAX_ITER = 4
        for _ in range(MAX_ITER):
            inputs_torch = [
                H_col, G_col, A_col, f_col, d_col, b_col,
                x_current.double().reshape(self.num_envs, -1, 1).contiguous(),
                s_current.double().reshape(self.num_envs, -1, 1).contiguous(),
                z_current.double().reshape(self.num_envs, -1, 1).contiguous(),
                y_current.double().reshape(self.num_envs, -1, 1).contiguous()
            ]
            
            # Measure solve time
            torch.cuda.synchronize()
            solve_iter_start = time()
            self.qp_solver.evaluate(inputs_torch)
            torch.cuda.synchronize()
            solve_iter_time = time() - solve_iter_start
            
            # Get outputs for all environments (already on GPU as PyTorch tensors)
            x_new = self.qp_solver.outputs_sparse[0].reshape(self.num_envs, -1)  # (self.num_envs, nz)
            s_new = self.qp_solver.outputs_sparse[1].reshape(self.num_envs, -1)  # (self.num_envs, num_ineq)
            z_new = self.qp_solver.outputs_sparse[2].reshape(self.num_envs, -1)  # (self.num_envs, num_ineq)
            y_new = self.qp_solver.outputs_sparse[3].reshape(self.num_envs, -1)  # (self.num_envs, num_eq)
            
            # Always use the solver output (trust it completely)
            x_current = x_new.clone().detach()
            s_current = s_new.clone().detach()
            z_current = z_new.clone().detach()
            y_current = y_new.clone().detach()
            
        
        sol = x_current.clone().detach()  # (num_envs, nz) - Keep double precision
        
        solve_time = time() - solve_start
        total_time = qp_gen_time + solve_time
        
        # Print timing breakdown
        if self.cfg.print_solve_time:
            print(f"Timing breakdown:")
            print(f"  - Matrix generation: {1000*qp_gen_time:.4f} ms ({100*qp_gen_time/total_time:.1f}%)")
            print(f"  - QP solving: {1000*solve_time:.4f} ms ({100*solve_time/total_time:.1f}%)")
        
        
        # retrieve solution
        cost = torch.zeros(self.num_envs, device=self.device)  # for now, ignore retrieving cost
        u_control = sol[:, nx*self.horizon_length:nx*self.horizon_length+nu]  # (num_envs, 12)
        
        # solutions are in global coordinate
        left_grf = u_control[:, :3].float()  # (num_envs, 3) - Convert to float32
        right_grf = u_control[:, 3:6].float()  # (num_envs, 3) - Convert to float32
        left_grm = u_control[:, 6:9].float()  # (num_envs, 3) - Convert to float32
        right_grm = u_control[:, 9:12].float()  # (num_envs, 3) - Convert to float32
        left_grm[:, 0] = 0.0
        right_grm[:, 0] = 0.0
        
        # transform wrench to body frame
        left_grf_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ left_grf.unsqueeze(-1)).squeeze(-1) # (num_envs, 3)
        left_grm_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ left_grm.unsqueeze(-1)).squeeze(-1) # (num_envs, 3)
        right_grf_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ right_grf.unsqueeze(-1)).squeeze(-1) # (num_envs, 3)
        right_grm_body = (self.state_estimate_data.rotation_body.transpose(1, 2) @ right_grm.unsqueeze(-1)).squeeze(-1) # (num_envs, 3)
       
        foot_wrench = torch.cat([-left_grf_body, -left_grm_body, -right_grf_body, -right_grm_body], dim=1).reshape(-1, 2, 6) # (num_envs, 2, 6)
        foot_wrench = foot_wrench.float()

        return foot_wrench, cost