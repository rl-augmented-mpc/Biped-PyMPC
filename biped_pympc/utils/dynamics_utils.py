import torch

def create_srbd_qp_matrices(
    R_body_flat,         # shape: (B, T*9)
    I_world_inv_flat,    # shape: (B, T*9)
    left_foot_pos_skew_flat,  # shape: (B, T*9)
    right_foot_pos_skew_flat, # shape: (B, T*9)
    contact_table,       # shape: (B, T*2)
    x_ref,               # shape: (B, T*13)
    Q,                   # shape: (13,) -- state weight (diagonal)
    R,                   # shape: (12,) -- control weight (diagonal)
    x0,                  # shape: (B, 13) initial state
    m,                   # scalar mass
    mu,                  # scalar friction coefficient
    dt_mpc,           # scalar timestep
):
    # parameters
    f_max = 500.0 # maximum z GRF
    # Constants
    num_x = 13
    num_u = 12
    num_dynamics_eq = 13    # equality constraints (dynamics) per time step
    num_eq = num_dynamics_eq+2 # dynamics + moment-x + zmp box constraints
    
    num_ineq_state = 0 # state inequality constraints per time step
    num_ineq_input = 16  # input inequality constraints per time step
    num_ineq = num_ineq_state + num_ineq_input # total inequality constraints per time step

    # Determine batch size B and horizon T from input sizes.
    B = R_body_flat.shape[0]
    T = R_body_flat.shape[1] // 9  # each time step has 9 entries (3x3 matrix)

    device = R_body_flat.device
    dtype = R_body_flat.dtype

    # Reshape flat inputs:
    R_body = R_body_flat.view(B, T, 3, 3)               # (B, T, 3, 3)
    I_world_inv = I_world_inv_flat.view(B, T, 3, 3)      # (B, T, 3, 3)
    left_foot_pos_skew = left_foot_pos_skew_flat.view(B, T, 3, 3)
    right_foot_pos_skew = right_foot_pos_skew_flat.view(B, T, 3, 3)
    contact_table = contact_table.view(B, T, 2)          # (B, T, 2)
    # left_pos_foot = unskew_symmetric(left_foot_pos_skew)  # (B, T, 3)
    # right_pos_foot = unskew_symmetric(right_foot_pos_skew) # (B, T, 3)

    # ----------------------------
    # Build per-timestep state-space matrices
    # ----------------------------

    # Build A_horizon (B, T, 13, 13)
    A_horizon = torch.zeros(B, T, num_x, num_x, device=device, dtype=dtype)
    A_horizon[:, :, 0:3, 6:9] = R_body                         # orientation integration
    A_horizon[:, :, 3:6, 9:12] = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    A_horizon[:, :, 11, 12] = -9.81
    
    # Discretize via forward Euler: A = I + A*dt_mpc
    I_state = torch.eye(num_x, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, T, num_x, num_x)
    A_horizon = I_state + A_horizon * dt_mpc

    # Build B_horizon (B, T, 13, 12)
    B_horizon = torch.zeros(B, T, num_x, num_u, device=device, dtype=dtype)
    # Left/right foot torque contributions:
    B_horizon[:, :, 6:9, 0:3] = torch.matmul(I_world_inv, left_foot_pos_skew)
    B_horizon[:, :, 6:9, 3:6] = torch.matmul(I_world_inv, right_foot_pos_skew)
    # Direct force contributions:
    B_horizon[:, :, 6:9, 6:9] = I_world_inv
    B_horizon[:, :, 6:9, 9:12] = I_world_inv
    # Mapping from foot force to linear acceleration:
    I3 = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(B, T, 3, 3)
    B_horizon[:, :, 9:12, 0:3] = (1/m) * I3
    B_horizon[:, :, 9:12, 3:6] = (1/m) * I3
    
    # Discretize:
    B_horizon = B_horizon * dt_mpc

    # ----------------------------
    # Build cost matrices H and gradient f
    # ----------------------------

    # Build constant diagonal weight matrices
    Q_mat = torch.diag_embed(Q)    # (13,13)
    R_mat = torch.diag_embed(R)    # (12,12)
    # Construct block diagonal matrices using the Kronecker product.
    Q_block = torch.kron(torch.eye(T, device=device, dtype=dtype), Q_mat)  # (T*13, T*13)
    R_block = torch.kron(torch.eye(T, device=device, dtype=dtype), R_mat)  # (T*12, T*12)
    # Hessian H is block diagonal of [Q, R] blocks
    H = torch.zeros(B, T*(num_x + num_u), T*(num_x + num_u), device=device, dtype=dtype)
    H[:, :T*num_x, :T*num_x] = Q_block
    H[:, T*num_x:, T*num_x:] = R_block

    # Build gradient f: first T*13 entries set from reference trajectory, then zeros.
    total_var = T * (num_x + num_u)
    f = torch.zeros(B, T * (num_x + num_u), 1, device=device, dtype=dtype)
    f[:, :T*num_x, :] = -Q_block @ x_ref.unsqueeze(2)

    # ----------------------------
    # Build equality constraints: A_block * [X; U] = b_block
    # ----------------------------

    # A_block has shape (B, T*13, T*(13+12))
    A_block_extra = torch.zeros(B, T*(num_eq - num_dynamics_eq), total_var, device=device, dtype=dtype)
    b_block_extra = torch.zeros(B, T*(num_eq - num_dynamics_eq), 1, device=device, dtype=dtype)
    
    A_block = torch.zeros(B, T*num_dynamics_eq, total_var, device=device, dtype=dtype)
    b_block = torch.zeros(B, T*num_dynamics_eq, 1, device=device, dtype=dtype)

    # For each time step i, assign:
    #   X_i block: identity, and U_i block: -B_horizon[:, i]
    for i in range(T):
        r_start = i * num_dynamics_eq
        r_end = (i+1) * num_dynamics_eq
        
        # State block: columns corresponding to X_i
        c_state_start = i * num_x
        c_state_end = (i+1) * num_x
        A_block[:, r_start:r_end, c_state_start:c_state_end] = \
            torch.eye(num_x, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        
        # Control block: columns corresponding to U_i
        c_ctrl_start = T * num_x + i * num_u
        c_ctrl_end = T * num_x + (i+1) * num_u
        A_block[:, r_start:r_end, c_ctrl_start:c_ctrl_end] = -B_horizon[:, i]
    
    # For i = 1,...,T-1 add the coupling from previous state: -A_horizon[:, i]
    for i in range(1, T):
        r_start = i * num_dynamics_eq
        r_end = (i+1) * num_dynamics_eq
        c_prev_state_start = (i-1) * num_x
        c_prev_state_end = i * num_x
        A_block[:, r_start:r_end, c_prev_state_start:c_prev_state_end] = -A_horizon[:, i]
    
    # Set the initial condition: b_block (first 13 rows) = A_horizon[:,0] @ x0.
    x0_ = x0.view(B, num_x, 1)
    b_block[:, :num_x, :] = torch.bmm(A_horizon[:, 0], x0_)
    
    # other constraints
    for i in range(T):
        c_ctrl_start = T * num_x + i * num_u
        c_ctrl_end = T * num_x + (i+1) * num_u
        A_block_extra[:, (num_eq-num_dynamics_eq)*i, c_ctrl_start+6] = 1.0 # Mx left
        A_block_extra[:, (num_eq-num_dynamics_eq)*i+1, c_ctrl_start+9] = 1.0 # Mx right
        
    
    A_block_total = torch.cat([A_block, A_block_extra], dim=1)
    b_block_total = torch.cat([b_block, b_block_extra], dim=1)
    
    # ----------------------------
    # Build inequality constraints: G_block * z <= d_block
    # ----------------------------
    
    G_input = torch.zeros(num_ineq_input, num_u, device=device, dtype=dtype)
    # Left foot force constraints
    G_input[0, 0] = -1
    G_input[0, 2] = -mu
    G_input[1, 0] = 1
    G_input[1, 2] = -mu
    G_input[2, 1] = -1
    G_input[2, 2] = -mu
    G_input[3, 1] = 1
    G_input[3, 2] = -mu
    
    # Left foot moment constraints
    lt_vec = torch.zeros(3, 1, device=device, dtype=dtype)
    lt_vec[2, 0] = -0.07  # lt
    lh_vec = torch.zeros(3, 1, device=device, dtype=dtype)
    lh_vec[2, 0] = 0.04  # lh
    Mx_sel = torch.zeros(3, 1, device=device, dtype=dtype)
    Mx_sel[0, 0] = 1
    My_sel = torch.zeros(3, 1, device=device, dtype=dtype)
    My_sel[1, 0] = 1
    
    # For left foot (foot rotation assumed identity)
    # [0, 0, -lt]*F -  [0, 1, 0]*M <= 0 
    # -[0, 0, lh]*F + [0, 1, 0]*M <= 0
    G_input[4, 0:3] = lt_vec.t() 
    G_input[4, 6:9] = -My_sel.t()
    G_input[5, 0:3] = -lh_vec.t()
    G_input[5, 6:9] = My_sel.t()
    # Left foot z force limit
    G_input[6, 2] = -1
    G_input[7, 2] = 1
    
    # Right foot force constraints
    G_input[8, 3] = -1
    G_input[8, 5] = -mu
    G_input[9, 3] = 1
    G_input[9, 5] = -mu
    G_input[10, 4] = -1
    G_input[10, 5] = -mu
    G_input[11, 4] = 1
    G_input[11, 5] = -mu
    # Right foot moment constraints
    G_input[12, 3:6] = lt_vec.t()
    G_input[12, 9:12] = -My_sel.t()
    G_input[13, 3:6] = -lh_vec.t()
    G_input[13, 9:12] = My_sel.t()
    # Right foot z force limit
    G_input[14, 5] = -1
    G_input[15, 5] = 1
    
    
    # Build full inequality constraint block.
    # For each time step, the inequality only acts on the control variables.
    # Build block diagonal G_block_control of shape (T*16, T*12):
    G_block = torch.zeros(B, T * num_ineq, T * (num_x + num_u), device=device, dtype=dtype)
    G_block[:, T*num_ineq_state:, T * num_x:] = torch.kron(torch.eye(T, device=device, dtype=dtype), G_input.unsqueeze(0).repeat(B, 1, 1))

    # Build d_block by repeating d_vec_input T times (shape becomes (B, T*16, 1))
    d_vec_input = torch.zeros(num_ineq_input, 1, device=device, dtype=dtype)
    d_input_block = d_vec_input.repeat(T, 1).unsqueeze(0).expand(B, -1, 1).clone()
    
    # apply contact table to d_block
    for i in range(T):
        d_input_block[:, i*num_ineq_input+7, :] = f_max * contact_table[:, i, 0].unsqueeze(1)
        d_input_block[:, i*num_ineq_input+15, :] = f_max * contact_table[:, i, 1].unsqueeze(1)
    d_block = d_input_block
    
    # column vectors into 1D
    f = f.squeeze(2)
    b_block_total = b_block_total.squeeze(2)
    d_block = d_block.squeeze(2)
    
    # ----------------------------
    # Return outputs (note: inequality matrices are negated to match the original output)
    # ----------------------------
    return H, f, A_block_total, b_block_total, G_block, d_block