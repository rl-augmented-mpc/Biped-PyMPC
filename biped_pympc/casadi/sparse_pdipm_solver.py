import casadi as ca
import numpy as np

def sparse_pdipm_solver_ccs(nz, num_eq, num_ineq, Q_row, Q_colptr, A_row, A_colptr, G_row, G_colptr, num_iters=100, tol=1e-8, beta=1e-8):
    """
    Symbolic PDIPM QP solver using CCS representation for Q, G, A with SPARSE KKT system.
    Sparsity pattern (row indices, col pointers) must be fixed and numeric.
    Only the values and variables are symbolic.
    Returns: CasADi function that takes (Q_val, G_val, A_val, f, h, b, x_init) and returns (x, residuals_mat)
    """
    nnz_Q = Q_colptr[-1]
    nnz_A = A_colptr[-1]
    nnz_G = G_colptr[-1]

    # Symbolic inputs as vectors of nonzeros
    Q_val = ca.SX.sym('Q_val', nnz_Q)
    G_val = ca.SX.sym('G_val', nnz_G)
    A_val = ca.SX.sym('A_val', nnz_A)
    f = ca.SX.sym('f', nz)
    h = ca.SX.sym('h', num_ineq)
    b = ca.SX.sym('b', num_eq)
    x_init = ca.SX.sym('x_init', nz)

    # Reconstruct sparse SX matrices from nonzero vectors
    Q_mat = sx_from_ccs(Q_val, Q_row, Q_colptr, (nz, nz))
    G_mat = sx_from_ccs(G_val, G_row, G_colptr, (num_ineq, nz))
    A_mat = sx_from_ccs(A_val, A_row, A_colptr, (num_eq, nz))

    # === Initialization ===
    x = x_init
    Gx = G_mat @ x
    s = ca.fmax(h - Gx, 1.0)  # strictly positive slacks
    z = ca.SX.ones(num_ineq)  # strictly positive duals for inequalities
    y = ca.SX.zeros(num_eq, 1)
    e = ca.SX.ones(num_ineq)  # vector of ones (for complementarity)

    # === Collect residuals for all iterations ===
    # residuals = []

    for it in range(num_iters):
        # === Compute KKT residuals ===
        Qx = Q_mat @ x
        Gx = G_mat @ x
        Ax = A_mat @ x
        if num_eq == 0:
            rx = Qx + f + G_mat.T @ z  # No equality constraints
            re = ca.SX.zeros(0)
        else:
            rx = Qx + f + G_mat.T @ z + A_mat.T @ y
            re = Ax - b
        rs = Gx + s - h
        # rc_vec = s * z
        mu = ca.dot(s, z) / num_ineq
        # Store residuals
        # rx_norm = ca.norm_2(ca.vec(rx))
        # rs_norm = ca.norm_2(ca.vec(rs))
        # re_norm = ca.norm_2(ca.vec(re))
        # rc_norm = ca.norm_2(ca.vec(rc_vec - mu * e))
        # residuals.append(ca.vertcat(rx_norm, rs_norm, re_norm, rc_norm, mu))

        # === Build sparse KKT system ===
        # Diagonal matrices for slacks and duals
        # S = ca.diag(s)
        Z = ca.diag(z)
        S_inv = ca.diag(1.0 / s)
        S_inv_Z = S_inv @ Z
        
        # Build sparse KKT system
        # Variable order: [dx, ds, dz, dy]
        # KKT matrix structure with regularization δI:
        #   [ Q+βI   0       G^T  A^T ]
        #   [ 0      S^{-1}Z+δI I    0  ]
        #   [ G      I       -δI  0  ]
        #   [ A      0       0    -δI ]
        
        total_dim = nz + num_ineq + num_ineq + num_eq
        KKT = ca.SX(total_dim, total_dim)
        
        # Regularization parameter δ
        delta = 1e-8
        
        # Top-left block: Q + βI
        KKT[:nz, :nz] = Q_mat + beta * ca.SX_eye(nz)
        
        # Top-right blocks: G^T and A^T
        KKT[:nz, nz+num_ineq:nz+2*num_ineq] = G_mat.T
        if num_eq > 0:
            KKT[:nz, nz+2*num_ineq:] = A_mat.T
        
        # Middle blocks: S^{-1}Z + δI and I
        KKT[nz:nz+num_ineq, nz:nz+num_ineq] = S_inv_Z + delta * ca.SX_eye(num_ineq)
        KKT[nz:nz+num_ineq, nz+num_ineq:nz+2*num_ineq] = ca.SX_eye(num_ineq)
        
        # Bottom-left blocks: G and A
        KKT[nz+num_ineq:nz+2*num_ineq, :nz] = G_mat
        KKT[nz+num_ineq:nz+2*num_ineq, nz:nz+num_ineq] = ca.SX_eye(num_ineq)
        if num_eq > 0:
            KKT[nz+2*num_ineq:, :nz] = A_mat
        
        # Bottom-right blocks: -δI terms
        KKT[nz+num_ineq:nz+2*num_ineq, nz+num_ineq:nz+2*num_ineq] = KKT[nz+num_ineq:nz+2*num_ineq, nz+num_ineq:nz+2*num_ineq] - delta * ca.SX_eye(num_ineq)
        if num_eq > 0:
            KKT[nz+2*num_ineq:, nz+2*num_ineq:] = -delta * ca.SX_eye(num_eq)

        # === Solve KKT system ===
        # Assemble right-hand side (scale complementarity row)
        rhs = ca.vertcat(
            -rx.reshape((-1, 1)),
            -(S_inv @ (s * z)).reshape((-1, 1)),
            -rs.reshape((-1, 1)),
            -re.reshape((-1, 1))
        )

        # Solve KKT system with better numerical handling
        D, L, perm = ca.ldl(KKT)
        sol_aff = ca.ldl_solve(rhs, D, L, perm)
            
        dx_aff = sol_aff[:nz]
        ds_aff = sol_aff[nz:nz+num_ineq]
        dz_aff = sol_aff[nz+num_ineq:nz+2*num_ineq]
        dy_aff = sol_aff[nz+2*num_ineq:]
        
        # === Compute step sizes for affine direction ===
        alpha_candidates_s = [ca.if_else(ds_aff[i] < 0, -s[i]/ds_aff[i], 1.0) for i in range(num_ineq)]
        alpha_aff_pri = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_s)))
        alpha_candidates_z = [ca.if_else(dz_aff[i] < 0, -z[i]/dz_aff[i], 1.0) for i in range(num_ineq)]
        alpha_aff_dual = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_z)))
        
        # === Add safeguards for step sizes ===
        alpha_aff_pri = ca.fmax(alpha_aff_pri, 1e-12)  # Ensure positive
        alpha_aff_dual = ca.fmax(alpha_aff_dual, 1e-12)  # Ensure positive
        
        # === Mehrotra predictor-corrector (corrector step) ===
        # Compute the affine step's predicted mu and centering parameter sigma
        s_aff = s + alpha_aff_pri * ds_aff
        z_aff = z + alpha_aff_dual * dz_aff
        mu_aff = ca.dot(s_aff, z_aff) / num_ineq
        sigma = (mu_aff / mu) ** 3
        
        # Corrector right-hand side for centering and corrector directions
        rc_corr = s * z + ds_aff * dz_aff - sigma * mu * e
        rhs_corr = ca.vertcat(ca.SX.zeros(nz), -S_inv @ rc_corr, ca.SX.zeros(num_ineq), ca.SX.zeros(num_eq))
        
        # Solve corrector system using the same KKT matrix
        # D, L, perm = ca.ldl(KKT)
        sol_corr = ca.ldl_solve(rhs_corr, D, L, perm)
        dx_corr = sol_corr[:nz]
        ds_corr = sol_corr[nz:nz+num_ineq]
        dz_corr = sol_corr[nz+num_ineq:nz+2*num_ineq]
        dy_corr = sol_corr[nz+2*num_ineq:]
        
        # === Combine affine and corrector directions ===
        dx = dx_aff + dx_corr
        ds = ds_aff + ds_corr
        dz = dz_aff + dz_corr
        dy = dy_aff + dy_corr
        
        # === Compute step sizes for combined direction ===
        alpha_candidates_s_comb = [ca.if_else(ds[i] < 0, -s[i]/ds[i], 1.0) for i in range(num_ineq)]
        alpha_pri = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_s_comb)))
        alpha_candidates_z_comb = [ca.if_else(dz[i] < 0, -z[i]/dz[i], 1.0) for i in range(num_ineq)]
        alpha_dual = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_z_comb)))
        
        # === Add safeguards for final step sizes ===
        alpha_pri = ca.fmax(alpha_pri, 1e-12)  # Ensure positive
        alpha_dual = ca.fmax(alpha_dual, 1e-12)  # Ensure positive
        
        # === Update primal and dual variables ===
        x = x + alpha_pri * dx
        s = s + alpha_pri * ds
        z = z + alpha_dual * dz
        y = y + alpha_dual * dy
        
        # === Add safeguards for variable updates ===
        s = ca.fmax(s, 1e-8)  # Ensure slacks remain positive
        z = ca.fmax(z, 1e-8)  # Ensure duals remain positive
        
        # === Clamp z to maintain positivity ===
        z = ca.fmax(z, 1e-8)

    # === Stack residuals into a matrix (shape: 5 x num_iters) ===
    # residuals_mat = ca.horzcat(*residuals)
    
    return ca.Function('sparse_pdipm_solver',
        [Q_val, G_val, A_val, f, h, b, x_init],
        [x])


def sparse_pdipm_single_iteration_ccs(nz, num_eq, num_ineq, Q_row, Q_colptr, A_row, A_colptr, G_row, G_colptr, beta=1e-8):
    """
    Symbolic PDIPM QP solver that performs a SINGLE iteration.
    EXACT COPY of multiple iteration logic - no differences.
    Returns: CasADi function that takes (Q_val, G_val, A_val, f, h, b, x, s, z, y) and returns (x_new, s_new, z_new, y_new, residuals, mu)
    """
    nnz_Q = Q_colptr[-1]
    nnz_A = A_colptr[-1]
    nnz_G = G_colptr[-1]

    # Symbolic inputs as vectors of nonzeros
    Q_val = ca.SX.sym('Q_val', nnz_Q)
    G_val = ca.SX.sym('G_val', nnz_G)
    A_val = ca.SX.sym('A_val', nnz_A)
    f = ca.SX.sym('f', nz)
    h = ca.SX.sym('h', num_ineq)
    b = ca.SX.sym('b', num_eq)
    
    # Current iterate variables
    x = ca.SX.sym('x', nz)
    s = ca.SX.sym('s', num_ineq)
    z = ca.SX.sym('z', num_ineq)
    y = ca.SX.sym('y', num_eq)

    # Reconstruct sparse SX matrices from nonzero vectors
    Q_mat = sx_from_ccs(Q_val, Q_row, Q_colptr, (nz, nz))
    G_mat = sx_from_ccs(G_val, G_row, G_colptr, (num_ineq, nz))
    A_mat = sx_from_ccs(A_val, A_row, A_colptr, (num_eq, nz))

    # === Compute KKT residuals (EXACT COPY) ===
    Qx = Q_mat @ x
    Gx = G_mat @ x
    Ax = A_mat @ x
    if num_eq == 0:
        rx = Qx + f + G_mat.T @ z  # No equality constraints
        re = ca.SX.zeros(0)
    else:
        rx = Qx + f + G_mat.T @ z + A_mat.T @ y
        re = Ax - b
    rs = Gx + s - h
    mu = ca.dot(s, z) / num_ineq

    # === Build sparse KKT system (EXACT COPY) ===
    # Diagonal matrices for slacks and duals
    Z = ca.diag(z)
    S_inv = ca.diag(1.0 / s)
    S_inv_Z = S_inv @ Z
    
    # Build sparse KKT system
    # Variable order: [dx, ds, dz, dy]
    total_dim = nz + num_ineq + num_ineq + num_eq
    KKT = ca.SX(total_dim, total_dim)
    
    # Regularization parameter δ (EXACT COPY)
    delta = 1e-8
    
    # Top-left block: Q + βI
    KKT[:nz, :nz] = Q_mat + beta * ca.SX_eye(nz)
    
    # Top-right blocks: G^T and A^T
    KKT[:nz, nz+num_ineq:nz+2*num_ineq] = G_mat.T
    if num_eq > 0:
        KKT[:nz, nz+2*num_ineq:] = A_mat.T
    
    # Middle blocks: S^{-1}Z + δI and I
    KKT[nz:nz+num_ineq, nz:nz+num_ineq] = S_inv_Z + delta * ca.SX_eye(num_ineq)
    KKT[nz:nz+num_ineq, nz+num_ineq:nz+2*num_ineq] = ca.SX_eye(num_ineq)
    
    # Bottom-left blocks: G and A
    KKT[nz+num_ineq:nz+2*num_ineq, :nz] = G_mat
    KKT[nz+num_ineq:nz+2*num_ineq, nz:nz+num_ineq] = ca.SX_eye(num_ineq)
    if num_eq > 0:
        KKT[nz+2*num_ineq:, :nz] = A_mat
    
    # Bottom-right blocks: -δI terms
    KKT[nz+num_ineq:nz+2*num_ineq, nz+num_ineq:nz+2*num_ineq] = KKT[nz+num_ineq:nz+2*num_ineq, nz+num_ineq:nz+2*num_ineq] - delta * ca.SX_eye(num_ineq)
    if num_eq > 0:
        KKT[nz+2*num_ineq:, nz+2*num_ineq:] = -delta * ca.SX_eye(num_eq)

    # === Solve KKT system (EXACT COPY) ===
    # Assemble right-hand side (scale complementarity row)
    rhs = ca.vertcat(
        -rx.reshape((-1, 1)),
        -(S_inv @ (s * z)).reshape((-1, 1)),
        -rs.reshape((-1, 1)),
        -re.reshape((-1, 1))
    )

    # Solve KKT system with better numerical handling
    D, L, perm = ca.ldl(KKT)
    sol_aff = ca.ldl_solve(rhs, D, L, perm)
        
    dx_aff = sol_aff[:nz]
    ds_aff = sol_aff[nz:nz+num_ineq]
    dz_aff = sol_aff[nz+num_ineq:nz+2*num_ineq]
    dy_aff = sol_aff[nz+2*num_ineq:]
    
    # === Compute step sizes for affine direction (EXACT COPY) ===
    alpha_candidates_s = [ca.if_else(ds_aff[i] < 0, -s[i]/ds_aff[i], 1.0) for i in range(num_ineq)]
    alpha_aff_pri = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_s)))
    alpha_candidates_z = [ca.if_else(dz_aff[i] < 0, -z[i]/dz_aff[i], 1.0) for i in range(num_ineq)]
    alpha_aff_dual = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_z)))
    
    # === Add safeguards for step sizes (EXACT COPY) ===
    alpha_aff_pri = ca.fmax(alpha_aff_pri, 1e-12)  # Ensure positive
    alpha_aff_dual = ca.fmax(alpha_aff_dual, 1e-12)  # Ensure positive
    
    # === Mehrotra predictor-corrector (corrector step) (EXACT COPY) ===
    # Compute the affine step's predicted mu and centering parameter sigma
    s_aff = s + alpha_aff_pri * ds_aff
    z_aff = z + alpha_aff_dual * dz_aff
    mu_aff = ca.dot(s_aff, z_aff) / num_ineq
    sigma = (mu_aff / mu) ** 3  # EXACT COPY - cubic sigma
    
    # Corrector right-hand side for centering and corrector directions
    e = ca.SX.ones(num_ineq)  # vector of ones (for complementarity)
    rc_corr = s * z + ds_aff * dz_aff - sigma * mu * e
    rhs_corr = ca.vertcat(ca.SX.zeros(nz), -S_inv @ rc_corr, ca.SX.zeros(num_ineq), ca.SX.zeros(num_eq))
    
    # Solve corrector system using the same KKT matrix
    sol_corr = ca.ldl_solve(rhs_corr, D, L, perm)
    dx_corr = sol_corr[:nz]
    ds_corr = sol_corr[nz:nz+num_ineq]
    dz_corr = sol_corr[nz+num_ineq:nz+2*num_ineq]
    dy_corr = sol_corr[nz+2*num_ineq:]
    
    # === Combine affine and corrector directions (EXACT COPY) ===
    dx = dx_aff + dx_corr
    ds = ds_aff + ds_corr
    dz = dz_aff + dz_corr
    dy = dy_aff + dy_corr
    
    # === Compute step sizes for combined direction (EXACT COPY) ===
    alpha_candidates_s_comb = [ca.if_else(ds[i] < 0, -s[i]/ds[i], 1.0) for i in range(num_ineq)]
    alpha_pri = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_s_comb)))
    alpha_candidates_z_comb = [ca.if_else(dz[i] < 0, -z[i]/dz[i], 1.0) for i in range(num_ineq)]
    alpha_dual = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_z_comb)))
    
    # === Add safeguards for final step sizes (EXACT COPY) ===
    alpha_pri = ca.fmax(alpha_pri, 1e-12)  # Ensure positive
    alpha_dual = ca.fmax(alpha_dual, 1e-12)  # Ensure positive
    
    # === Update primal and dual variables (EXACT COPY) ===
    x_new = x + alpha_pri * dx
    s_new = s + alpha_pri * ds
    z_new = z + alpha_dual * dz
    y_new = y + alpha_dual * dy
    
    # === Add safeguards for variable updates (EXACT COPY) ===
    s_new = ca.fmax(s_new, 1e-8)  # Ensure slacks remain positive
    z_new = ca.fmax(z_new, 1e-8)  # Ensure duals remain positive
    
    # === Clamp z to maintain positivity (EXACT COPY) ===
    z_new = ca.fmax(z_new, 1e-8)

    # === Compute final mu (EXACT COPY - no manual management) ===
    mu_new = ca.dot(s_new, z_new) / num_ineq

    # === Compute residuals for output ===
    rx_norm = ca.norm_2(ca.vec(rx))
    rs_norm = ca.norm_2(ca.vec(rs))
    re_norm = ca.norm_2(ca.vec(re))
    residuals = ca.vertcat(rx_norm, rs_norm, re_norm, mu_new)
    
    return ca.Function('sparse_pdipm_single_iteration',
        [Q_val, G_val, A_val, f, h, b, x, s, z, y],
        [x_new, s_new, z_new, y_new, residuals, mu_new])

def sparse_pdipm_multiple_iterations(nz, num_eq, num_ineq, Q_row, Q_colptr, A_row, A_colptr, G_row, G_colptr, num_iters=10, tol=1e-8, beta=1e-8):
    """
    Symbolic PDIPM QP solver that performs multiple iterations.
    Returns: CasADi function that takes (Q_val, G_val, A_val, f, h, b, x, s, z, y) and returns (x_new, s_new, z_new, y_new, residuals, mu)
    """
    nnz_Q = Q_colptr[-1]
    nnz_A = A_colptr[-1]
    nnz_G = G_colptr[-1]

    # Symbolic inputs as vectors of nonzeros
    Q_val = ca.SX.sym('Q_val', nnz_Q)
    G_val = ca.SX.sym('G_val', nnz_G)
    A_val = ca.SX.sym('A_val', nnz_A)
    f = ca.SX.sym('f', nz)
    h = ca.SX.sym('h', num_ineq)
    b = ca.SX.sym('b', num_eq)
    
    # Current iterate variables
    x = ca.SX.sym('x', nz)
    s = ca.SX.sym('s', num_ineq)
    z = ca.SX.sym('z', num_ineq)
    y = ca.SX.sym('y', num_eq)

    # Reconstruct sparse SX matrices from nonzero vectors
    Q_mat = sx_from_ccs(Q_val, Q_row, Q_colptr, (nz, nz))
    G_mat = sx_from_ccs(G_val, G_row, G_colptr, (num_ineq, nz))
    A_mat = sx_from_ccs(A_val, A_row, A_colptr, (num_eq, nz))

    # === Multiple iterations without reassigning symbolic variables ===
    x_current = x
    s_current = s
    z_current = z
    y_current = y
    
    for i in range(num_iters):
        Qx = Q_mat @ x_current
        Gx = G_mat @ x_current
        Ax = A_mat @ x_current
        if num_eq == 0:
            rx = Qx + f + G_mat.T @ z_current  # No equality constraints
            re = ca.SX.zeros(0)
        else:
            rx = Qx + f + G_mat.T @ z_current + A_mat.T @ y_current
            re = Ax - b
        rs = Gx + s_current - h
        mu = ca.dot(s_current, z_current) / num_ineq

        # === Build sparse KKT system ===
        # Diagonal matrices for slacks and duals
        Z = ca.diag(z_current)
        S_inv = ca.diag(1.0 / s_current)
        S_inv_Z = S_inv @ Z
        
        # Build sparse KKT system
        # Variable order: [dx, ds, dz, dy]
        total_dim = nz + num_ineq + num_ineq + num_eq
        KKT = ca.SX(total_dim, total_dim)
        
        # Regularization parameter δ
        delta = 1e-8
        
        # Top-left block: Q + βI
        KKT[:nz, :nz] = Q_mat + beta * ca.SX_eye(nz)
        
        # Top-right blocks: G^T and A^T
        KKT[:nz, nz+num_ineq:nz+2*num_ineq] = G_mat.T
        if num_eq > 0:
            KKT[:nz, nz+2*num_ineq:] = A_mat.T
        
        # Middle blocks: S^{-1}Z + δI and I
        KKT[nz:nz+num_ineq, nz:nz+num_ineq] = S_inv_Z + delta * ca.SX_eye(num_ineq)
        KKT[nz:nz+num_ineq, nz+num_ineq:nz+2*num_ineq] = ca.SX_eye(num_ineq)
        
        # Bottom-left blocks: G and A
        KKT[nz+num_ineq:nz+2*num_ineq, :nz] = G_mat
        KKT[nz+num_ineq:nz+2*num_ineq, nz:nz+num_ineq] = ca.SX_eye(num_ineq)
        if num_eq > 0:
            KKT[nz+2*num_ineq:, :nz] = A_mat
        
        # Bottom-right blocks: -δI terms
        KKT[nz+num_ineq:nz+2*num_ineq, nz+num_ineq:nz+2*num_ineq] = KKT[nz+num_ineq:nz+2*num_ineq, nz+num_ineq:nz+2*num_ineq] - delta * ca.SX_eye(num_ineq)
        if num_eq > 0:
            KKT[nz+2*num_ineq:, nz+2*num_ineq:] = -delta * ca.SX_eye(num_eq)

        # === Solve KKT system ===
        # Assemble right-hand side (scale complementarity row)
        rhs = ca.vertcat(
            -rx.reshape((-1, 1)),
            -(S_inv @ (s_current * z_current)).reshape((-1, 1)),
            -rs.reshape((-1, 1)),
            -re.reshape((-1, 1))
        )

        # Solve KKT system with better numerical handling
        D, L, perm = ca.ldl(KKT)
        sol_aff = ca.ldl_solve(rhs, D, L, perm)
            
        dx_aff = sol_aff[:nz]
        ds_aff = sol_aff[nz:nz+num_ineq]
        dz_aff = sol_aff[nz+num_ineq:nz+2*num_ineq]
        dy_aff = sol_aff[nz+2*num_ineq:]
        
        # === Compute step sizes for affine direction ===
        alpha_candidates_s = [ca.if_else(ds_aff[i] < 0, -s_current[i]/ds_aff[i], 1.0) for i in range(num_ineq)]
        alpha_aff_pri = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_s)))
        alpha_candidates_z = [ca.if_else(dz_aff[i] < 0, -z_current[i]/dz_aff[i], 1.0) for i in range(num_ineq)]
        alpha_aff_dual = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_z)))
        
        # === Add safeguards for step sizes ===
        alpha_aff_pri = ca.fmax(alpha_aff_pri, 1e-12)  # Ensure positive
        alpha_aff_dual = ca.fmax(alpha_aff_dual, 1e-12)  # Ensure positive
        
        # === Mehrotra predictor-corrector (corrector step) ===
        # Compute the affine step's predicted mu and centering parameter sigma
        s_aff = s_current + alpha_aff_pri * ds_aff
        z_aff = z_current + alpha_aff_dual * dz_aff
        mu_aff = ca.dot(s_aff, z_aff) / num_ineq
        sigma = (mu_aff / mu) ** 3
        
        # Corrector right-hand side for centering and corrector directions
        e = ca.SX.ones(num_ineq)  # vector of ones (for complementarity)
        rc_corr = s_current * z_current + ds_aff * dz_aff - sigma * mu * e
        rhs_corr = ca.vertcat(ca.SX.zeros(nz), -S_inv @ rc_corr, ca.SX.zeros(num_ineq), ca.SX.zeros(num_eq))
        
        # Solve corrector system using the same KKT matrix
        sol_corr = ca.ldl_solve(rhs_corr, D, L, perm)
        dx_corr = sol_corr[:nz]
        ds_corr = sol_corr[nz:nz+num_ineq]
        dz_corr = sol_corr[nz+num_ineq:nz+2*num_ineq]
        dy_corr = sol_corr[nz+2*num_ineq:]
        
        # === Combine affine and corrector directions ===
        dx = dx_aff + dx_corr
        ds = ds_aff + ds_corr
        dz = dz_aff + dz_corr
        dy = dy_aff + dy_corr
        
        # === Compute step sizes for combined direction ===
        alpha_candidates_s_comb = [ca.if_else(ds[i] < 0, -s_current[i]/ds[i], 1.0) for i in range(num_ineq)]
        alpha_pri = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_s_comb)))
        alpha_candidates_z_comb = [ca.if_else(dz[i] < 0, -z_current[i]/dz[i], 1.0) for i in range(num_ineq)]
        alpha_dual = ca.fmin(1.0, 0.99 * ca.mmin(ca.vertcat(*alpha_candidates_z_comb)))
        
        # === Add safeguards for final step sizes ===
        alpha_pri = ca.fmax(alpha_pri, 1e-12)  # Ensure positive
        alpha_dual = ca.fmax(alpha_dual, 1e-12)  # Ensure positive
        
        # === Update primal and dual variables ===
        x_new = x_current + alpha_pri * dx
        s_new = s_current + alpha_pri * ds
        z_new = z_current + alpha_dual * dz
        y_new = y_current + alpha_dual * dy
        
        # === Add safeguards for variable updates ===
        s_new = ca.fmax(s_new, 1e-8)  # Ensure slacks remain positive
        z_new = ca.fmax(z_new, 1e-8)  # Ensure duals remain positive
        
        # === Clamp z to maintain positivity ===
        z_new = ca.fmax(z_new, 1e-8)

        # === Update current variables for next iteration ===
        x_current = x_new
        s_current = s_new
        z_current = z_new
        y_current = y_new

        # === Compute final mu ===
        mu_new = ca.dot(s_new, z_new) / num_ineq

        # === Compute residuals for output ===
        rx_norm = ca.norm_2(ca.vec(rx))
        rs_norm = ca.norm_2(ca.vec(rs))
        re_norm = ca.norm_2(ca.vec(re))
        residuals = ca.vertcat(rx_norm, rs_norm, re_norm, mu_new)
    
    return ca.Function('sparse_pdipm_multiple_iterations',
        [Q_val, G_val, A_val, f, h, b, x, s, z, y],
        [x_new, s_new, z_new, y_new, residuals, mu_new])


def initialize_pdipm_variables(nz, num_eq, num_ineq, x_init, G_mat, h):
    """
    Initialize variables for single-iteration PDIPM solver.
    
    Args:
        nz: number of variables
        num_eq: number of equality constraints
        num_ineq: number of inequality constraints
        x_init: initial guess for primal variables
        G_mat: inequality constraint matrix
        h: inequality constraint RHS
        
    Returns:
        x, s, z, y: initialized primal and dual variables
    """
    x = x_init.copy()
    Gx = G_mat @ x
    s = np.maximum(h - Gx, 1.0)  # strictly positive slacks
    z = np.ones(num_ineq)  # strictly positive duals for inequalities
    y = np.zeros(num_eq)  # duals for equality constraints
    
    return x, s, z, y


def get_ccs_from_sx(mat):
    """
    Extract CCS (Compressed Column Storage) entities from a CasADi SX matrix.
    Returns: (values, row_indices, col_pointers)
    """
    values = mat.nonzeros()
    col_pointers, row_indices = mat.sparsity().get_ccs()
    return np.array(values).astype(np.float64), row_indices, col_pointers


def sx_from_ccs(values, row_indices, col_pointers, shape):
    """
    Construct a CasADi SX matrix from CCS (Compressed Column Storage) entities.
    Args:
        values: list/array of nonzero values
        row_indices: list/array of row indices for each nonzero
        col_pointers: list/array of column pointers (length ncols+1)
        shape: (nrows, ncols)
    Returns:
        CasADi SX matrix of given shape
    """
    rows, cols = shape
    mat = ca.SX(rows, cols)
    for j in range(cols):
        col_start = col_pointers[j]
        col_end = col_pointers[j+1]
        for idx in range(col_start, col_end):
            i = row_indices[idx]
            val = values[idx]
            mat[i, j] = val
    return mat



if __name__ == "__main__":
    # Example: small 2x2 QP with CCS data
    nz = 5
    num_eq = 2
    num_ineq = 2
    # Q = [[2, 0], [0, 2]]
    Q = np.eye(nz) * 2.0  # (nz, nz) diagonal matrix
    # Make G a random sparse diagonal matrix (works if num_ineq == nz)
    G = np.zeros((num_ineq, nz))
    for i in range(min(num_ineq, nz)):
        G[i, i] = np.random.randn()
    # Make A a random sparse diagonal matrix (works if num_eq == nz)
    A = np.zeros((num_eq, nz))
    for i in range(min(num_eq, nz)):
        A[i, i] = np.random.randn()
    f = np.random.randn(nz)
    h = np.abs(np.random.randn(num_ineq)) + 1.0
    b = np.random.randn(num_eq)
    x_init = np.random.randn(nz)

    # Convert CasADi SX to CCS 
    Q_sx = ca.SX(ca.Sparsity.diag(nz))
    for i in range(nz):
        Q_sx[i, i] = 2.0
    if num_ineq == nz:
        G_sx = ca.SX(ca.Sparsity.diag(nz))
        for i in range(nz):
            G_sx[i, i] = G[i, i]
    else:
        G_sx = ca.SX(G)
    if num_eq == nz:
        A_sx = ca.SX(ca.Sparsity.diag(nz))
        for i in range(nz):
            A_sx[i, i] = A[i, i]
    else:
        A_sx = ca.SX(A)

    Q_val, Q_row, Q_colptr = get_ccs_from_sx(Q_sx)
    G_val, G_row, G_colptr = get_ccs_from_sx(G_sx)
    A_val, A_row, A_colptr = get_ccs_from_sx(A_sx)
    

    sparse_solver = sparse_pdipm_solver_ccs(
        nz, num_eq, num_ineq,
        Q_row, Q_colptr,
        A_row, A_colptr,
        G_row, G_colptr,
        num_iters=10
    )

    single_iter_solver = sparse_pdipm_single_iteration_ccs(
        nz, num_eq, num_ineq,
        Q_row, Q_colptr,
        A_row, A_colptr,
        G_row, G_colptr,
        beta=1e-8
    )   

    print("multiple_iter_solver")
    multiple_iter_solver = sparse_pdipm_multiple_iterations(
        nz, num_eq, num_ineq,
        Q_row, Q_colptr,
        A_row, A_colptr,
        G_row, G_colptr,
        num_iters=10
    )
    
    x_init, s, z, y = initialize_pdipm_variables(nz, num_eq, num_ineq, x_init, G, h)
    
    x_opt = sparse_solver(
        Q_val, G_val, A_val,
        f, h, b, x_init
    )
    x_opt_single_iter = single_iter_solver(
        Q_val, G_val, A_val,
        f, h, b, x_init, s, z, y
    )
    x_opt_multiple_iter = multiple_iter_solver(
        Q_val, G_val, A_val,
        f, h, b, x_init, s, z, y
    )

    print(x_opt_multiple_iter)
    print("__________________________________")

    # single_iter_solver.save('single_iter_solver.casadi')
    # print("Saved CasADi function to 'single_iter_solver.casadi'")

    

    # print("__________________________________")
    # print("Optimal solution x:", x_opt)
    # print("num instructions: ", sparse_solver.n_instructions())
    # print("Residuals (iter, rx, rs, re, rc, mu) for all iterations:")
    # print(f"residuals_mat type: {type(residuals_mat)}")
    # print(f"residuals_mat shape: {residuals_mat.shape}")
    # residuals_np = np.array(residuals_mat)
    # print(f"residuals_np type: {type(residuals_np)}")
    # print(f"residuals_np shape: {residuals_np.shape}")
    
    

    # print('Function input signatures:')
    # for i in range(sparse_solver.n_in()):
    #     print(f'Input {i}:', sparse_solver.name_in(i), sparse_solver.sparsity_in(i))