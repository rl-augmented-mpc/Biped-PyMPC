import os
from os import system
import casadi as ca
import numpy as np
from src.casadi.sparse_pdipm_solver import sparse_pdipm_solver_ccs

def to_float(x):
    try:
        return float(x.full().item())  # works if x is a casadi.DM
    except AttributeError:
        return float(x)  # already a float or something float-like

def analyze_kkt_matrix_numeric(KKT_np, it, nz, num_ineq, num_eq):
    """
    Analyze KKT matrix properties using actual numeric values.
    """
    try:
        # Check for NaN/Inf
        has_nan = np.any(np.isnan(KKT_np))
        has_inf = np.any(np.isinf(KKT_np))
        
        # Check symmetry
        is_symmetric = np.allclose(KKT_np, KKT_np.T, atol=1e-10)
        
        # Compute eigenvalues
        eigenvals = np.linalg.eigvals(KKT_np)
        min_eigenval = np.min(eigenvals)
        max_eigenval = np.max(eigenvals)
        
        # Check positive semi-definiteness
        is_psd = np.all(eigenvals >= -1e-10)
        
        # Check positive definiteness
        is_pd = np.all(eigenvals > 1e-10)
        
        # Condition number
        cond_num = np.linalg.cond(KKT_np)
        
        print(f"\n=== Iteration {it} KKT Matrix Analysis ===")
        print(f"Shape: {KKT_np.shape}")
        print(f"Has NaN: {has_nan}")
        print(f"Has Inf: {has_inf}")
        print(f"Is symmetric: {is_symmetric}")
        print(f"Min eigenvalue: {min_eigenval:.2e}")
        print(f"Max eigenvalue: {max_eigenval:.2e}")
        print(f"Is PSD: {is_psd}")
        print(f"Is PD: {is_pd}")
        print(f"Condition number: {cond_num:.2e}")
        
        # Analyze blocks
        total_dim = nz + num_ineq + num_ineq + num_eq
        
        # Top-left block (Q + βI)
        Q_block = KKT_np[:nz, :nz]
        Q_eigenvals = np.linalg.eigvals(Q_block)
        print(f"Q block min eigenval: {np.min(Q_eigenvals):.2e}")
        print(f"Q block max eigenval: {np.max(Q_eigenvals):.2e}")
        
        # Middle block (S^{-1}Z)
        S_inv_Z_block = KKT_np[nz:nz+num_ineq, nz:nz+num_ineq]
        S_inv_Z_eigenvals = np.linalg.eigvals(S_inv_Z_block)
        print(f"S_inv_Z block min eigenval: {np.min(S_inv_Z_eigenvals):.2e}")
        print(f"S_inv_Z block max eigenval: {np.max(S_inv_Z_eigenvals):.2e}")
        
        # Print some eigenvalues for debugging
        print(f"All eigenvalues: {eigenvals}")
        
        return {
            'has_nan': has_nan,
            'has_inf': has_inf,
            'is_symmetric': is_symmetric,
            'min_eigenval': min_eigenval,
            'max_eigenval': max_eigenval,
            'is_psd': is_psd,
            'is_pd': is_pd,
            'cond_num': cond_num,
            'eigenvals': eigenvals
        }
        
    except Exception as e:
        print(f"Error analyzing KKT matrix at iteration {it}: {e}")
        return None
    
if __name__ == "__main__":
    # 1. Load CasADi function
    casadi_fn_path = "hector_pytorch/casadi/function/srbd_qp_mat.casadi"
    qp_former = ca.Function.load(casadi_fn_path)
    # print("Loaded CasADi QP matrix function.")

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
        Q, R
    )

    # get CCS triplets 
    Hzz = H.nonzeros()
    H_col_point, H_row_ind = H.sparsity().get_ccs()

    Azz = A.nonzeros()
    A_col_point, A_row_ind = A.sparsity().get_ccs()

    Gzz = G.nonzeros()
    G_col_point, G_row_ind = G.sparsity().get_ccs()

    # print("QP matrices formed.")

    # Get dimensions
    nz = H.shape[0]  # number of variables
    num_eq = A.shape[0]  # number of equality constraints
    num_ineq = G.shape[0]  # number of inequality constraints

    print(f"Dimensions: nz={nz}, num_eq={num_eq}, num_ineq={num_ineq}")

    # Create the solver function using sparsity patterns
    max_iter = 10
    solver = sparse_pdipm_solver_ccs(
        nz, num_eq, num_ineq,
        H_row_ind, H_col_point,
        A_row_ind, A_col_point, 
        G_row_ind, G_col_point,
        num_iters=max_iter, tol=1e-4, beta=1e-6  # Much more conservative parameters for MPC
    )

    print("Number of instructions", solver.n_instructions())

    # Save the CasADi function
    function_path = os.path.join("hector_pytorch", "casadi", "function", f"mpc_pdipm_solver_{max_iter}.casadi")
    solver.save(function_path)
    print(f"Saved CasADi function to {function_path}")
    
    # copy file to cusadi directory too
    system(f"cp {function_path} {os.path.join('hector_pytorch/cusadi/src/casadi_functions', f'mpc_pdipm_solver_{max_iter}.casadi')}")
    


    # Now call the solver with actual values
    # Function expects: [Q_val, G_val, A_val, f, h, b, x_init]
    # where h is the inequality constraint RHS (d), and x_init is the full solution vector z

    # Convert lists to numpy arrays and then to column vectors
    Hzz_col = np.array(Hzz).reshape(-1, 1)
    Gzz_col = np.array(Gzz).reshape(-1, 1) 
    Azz_col = np.array(Azz).reshape(-1, 1)

    # Convert CasADi objects to numpy arrays and reshape
    f_col = np.array(f).reshape(-1, 1)
    d_col = np.array(d).reshape(-1, 1)  # h in function signature
    b_col = np.array(b).reshape(-1, 1)
    z_col = np.array(z).reshape(-1, 1)  # x_init should be full solution vector

    # Call solver and unpack results
    x_opt = solver(
        Hzz_col, Gzz_col, Azz_col,
        f_col, d_col, b_col, z_col
    )
    # print(x_opt.shape)

    # print("=== MPC Solver Results ===")
    # print(f"Optimal solution: {x_opt[12*horizon:12*horizon+12]}")
    # print(f"Optimal solution shape: {x_opt.shape}")
    # print(f"Optimal solution norm: {np.linalg.norm(x_opt)}")

    # # Check for NaN values in optimal solution
    # x_opt_np = np.array(x_opt)
    # print(f"Optimal solution contains NaN: {np.any(np.isnan(x_opt_np))}")
    # print(f"Optimal solution contains Inf: {np.any(np.isinf(x_opt_np))}")
    # print(f"Optimal solution min: {np.min(x_opt_np)}")
    # print(f"Optimal solution max: {np.max(x_opt_np)}")