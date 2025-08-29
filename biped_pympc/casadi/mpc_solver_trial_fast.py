import os
from os import system
import casadi as ca
import numpy as np
from biped_pympc.casadi.sparse_pdipm_solver import sparse_pdipm_single_iteration_ccs, sparse_pdipm_multiple_iterations, initialize_pdipm_variables

if __name__ == "__main__":

    # 1. Load CasADi function
    casadi_fn_path = os.path.join("biped_pympc", "casadi", "function", "srbd_qp_mat.casadi")
    qp_former = ca.Function.load(casadi_fn_path)

    # 2. Realistic input data (from srbd_constraints.py)
    horizon = 10
    num_x = 12
    num_u = 12

    # dynamics parameters
    dt = 0.001*40  # mpc sampling time
    m = 13.5
    mu = 0.5
    R_body = np.eye(3)
    I_world = np.array([[0.5413, 0.0, 0.0],
                        [0.0, 0.5200, 0.0],
                        [0.0, 0.0, 0.0691]])
    body_pos = np.array([0.0, 0.0, 0.5])
    left_foot_pos = np.array([0.1, 0.05, 0.0])
    right_foot_pos = np.array([0.1, -0.05, 0.0])
    contact_table = np.ones((horizon, 2))

    # MPC tracking weight
    Q = np.array([200, 500, 500, 500, 500, 500, 1, 1, 5, 1, 1, 5])
    R = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-5, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2])

    # linear and angular residual accelerations (to simulate model mismatch)
    residual_lin_accel = np.array([0.0, 0.0, 0.0])
    residual_ang_accel = np.array([0.0, 0.0, 0.0])

    # form realistic state and control
    z = np.zeros(((12+12)*horizon))
    for i in range(horizon):
        z[i*12+5] = 0.5  # z position
        z[12*horizon + i*12+2] = m*9.81/2  # left foot z force
        z[12*horizon + i*12+5] = m*9.81/2  # right foot z force
        z[12*horizon + i*12+6:12*horizon + i*12+9] = np.cross(left_foot_pos-body_pos, z[12*horizon + i*12:12*horizon + i*12+3])
        z[12*horizon + i*12+9:12*horizon + i*12+12] = np.cross(right_foot_pos-body_pos, z[12*horizon + i*12+3:12*horizon + i*12+6])

    x = z[:12*horizon]
    u = z[12*horizon:]

    # initial state
    x0 = np.zeros((num_x,))
    x0[5] = 0.55  # z position

    # reference trajectory
    x_ref = np.zeros((12*horizon))
    for i in range(horizon):
        x_ref[i*12+5] = 0.55  # desired z position

    # 3. Call CasADi function
    H, f, A, b, G, d = qp_former(
        x0, x, u, x_ref, dt, m, mu,
        R_body, I_world, body_pos,
        left_foot_pos, right_foot_pos, contact_table,
        Q, R, residual_lin_accel, residual_ang_accel
    )

    # get CCS triplets 
    Hzz = H.nonzeros()
    H_col_point, H_row_ind = H.sparsity().get_ccs()

    Azz = A.nonzeros()
    A_col_point, A_row_ind = A.sparsity().get_ccs()

    Gzz = G.nonzeros()
    G_col_point, G_row_ind = G.sparsity().get_ccs()

    print("QP matrices formed.")

    # Get dimensions
    nz = H.shape[0]  # number of variables
    num_eq = A.shape[0]  # number of equality constraints
    num_ineq = G.shape[0]  # number of inequality constraints

    print(f"Dimensions: nz={nz}, num_eq={num_eq}, num_ineq={num_ineq}")

    # # Create the single-iteration solver function using sparsity patterns
    # solver = sparse_pdipm_single_iteration_ccs(
    #     nz, num_eq, num_ineq,
    #     H_row_ind, H_col_point,
    #     A_row_ind, A_col_point, 
    #     G_row_ind, G_col_point,
    #     beta=1e-8
    # )

    # print("Number of instructions per iteration:", solver.n_instructions())

    # # Save the CasADi function with a descriptive name
    # function_path = os.path.join("biped_pympc", "casadi", "function", f'mpc_solver_{nz}v_{num_eq}eq_{num_ineq}ineq.casadi')
    # solver.save(function_path)
    # print(f"Saved CasADi function to '{function_path}'")
    # system(f"cp {function_path} {os.path.join('biped_pympc/cusadi/src/casadi_functions', f'mpc_solver_{nz}v_{num_eq}eq_{num_ineq}ineq.casadi')}")


    # Create the single-iteration solver function using sparsity patterns
    MAX_ITER = 5
    solver = sparse_pdipm_multiple_iterations(
        nz, num_eq, num_ineq,
        H_row_ind, H_col_point,
        A_row_ind, A_col_point, 
        G_row_ind, G_col_point,
        beta=1e-8, 
        num_iters=MAX_ITER,
    )

    print("Number of instructions:", solver.n_instructions())

    # Save the CasADi function with a descriptive name
    function_path = os.path.join("biped_pympc", "casadi", "function", f'mpc_multiple_iter_{MAX_ITER}_solver_{nz}v_{num_eq}eq_{num_ineq}ineq.casadi')
    solver.save(function_path)
    print(f"Saved CasADi function to '{function_path}'")
    system(f"cp {function_path} {os.path.join('biped_pympc/cusadi/src/casadi_functions', f'mpc_multiple_iter_{MAX_ITER}_solver_{nz}v_{num_eq}eq_{num_ineq}ineq.casadi')}")



    #### Solve toy problem ####
    # # Convert lists to numpy arrays and then to column vectors
    # Hzz_col = np.array(Hzz).reshape(-1, 1)
    # Gzz_col = np.array(Gzz).reshape(-1, 1) 
    # Azz_col = np.array(Azz).reshape(-1, 1)

    # # Convert CasADi objects to numpy arrays and reshape
    # f_col = np.array(f).reshape(-1, 1)
    # d_col = np.array(d).reshape(-1, 1)  # h in function signature
    # b_col = np.array(b).reshape(-1, 1)

    # # Initialize variables for the iterative solver
    # x_current = np.array(z).reshape(-1, 1)  # initial guess
    # G_mat_np = np.array(G)
    # h_np = d_col.flatten()

    # # Use helper function to initialize variables properly
    # x_current, s_current, z_current, y_current = initialize_pdipm_variables(
    #     nz, num_eq, num_ineq, 
    #     x_current.flatten(), G_mat_np, h_np
    # )

    # print("=== Single-Iteration MPC Solver Results ===")
    # print(f"Initial solution norm: {np.linalg.norm(x_current)}")

    # # External loop for PDIPM iterations
    # max_iters = 10
    # tol = 1e-8
    # converged = False

    # for iter_num in range(max_iters):
    #     # Call single iteration solver
    #     x_new, s_new, z_new, y_new, residuals, mu = solver(
    #         Hzz_col, Gzz_col, Azz_col,
    #         f_col, d_col, b_col, 
    #         x_current.reshape(-1, 1), s_current, z_current, y_current
    #     )
        
    #     # Convert CasADi outputs to numpy
    #     x_new_np = np.array(x_new).flatten()
    #     s_new_np = np.array(s_new).flatten()
    #     z_new_np = np.array(z_new).flatten()
    #     y_new_np = np.array(y_new).flatten()
    #     residuals_np = np.array(residuals).flatten()
    #     mu_np = float(mu)
        
    #     # Check convergence
    #     rx_norm = residuals_np[0]
    #     rs_norm = residuals_np[1]
    #     re_norm = residuals_np[2]
        
    #     print(f"Iteration {iter_num+1}: rx_norm={rx_norm:.2e}, rs_norm={rs_norm:.2e}, re_norm={re_norm:.2e}, mu={mu_np:.2e}")
        
    #     # Update current variables
    #     x_current = x_new_np
    #     s_current = s_new_np
    #     z_current = z_new_np
    #     y_current = y_new_np
        
    #     # Check convergence criteria
    #     if rx_norm < tol and rs_norm < tol and re_norm < tol and mu_np < tol:
    #         converged = True
    #         print(f"Converged at iteration {iter_num+1}")
    #         break

    # if not converged:
    #     print(f"Did not converge after {max_iters} iterations")

    # print(f"Final solution: {x_current[12*horizon:12*horizon+12]}")
    # print(f"Final solution shape: {x_current.shape}")
    # print(f"Final solution norm: {np.linalg.norm(x_current)}")

    # # Check for NaN values in optimal solution
    # print(f"Final solution contains NaN: {np.any(np.isnan(x_current))}")
    # print(f"Final solution contains Inf: {np.any(np.isinf(x_current))}")
    # print(f"Final solution min: {np.min(x_current)}")
    # print(f"Final solution max: {np.max(x_current)}")

    # print(f"\n=== Performance Summary ===")
    # print(f"Instructions per iteration: {solver.n_instructions()}")
    # print(f"Total iterations: {iter_num+1}")
    # print(f"Total instructions: {solver.n_instructions() * (iter_num+1)}")
    # print(f"Converged: {converged}") 