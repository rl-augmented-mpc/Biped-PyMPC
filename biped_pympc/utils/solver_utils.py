from typing import Tuple
import torch
# import cupy as cp

def qp_solver_torch(H:torch.Tensor, g:torch.Tensor, A:torch.Tensor, b:torch.Tensor, G:torch.Tensor, h:torch.Tensor, rho:float)->Tuple[torch.Tensor, torch.Tensor]: 
    """
    Arguments:
    H: torch.Tensor of shape (batch_size, nz, nz)
    g: torch.Tensor of shape (batch_size, nz)
    A: torch.Tensor of shape (batch_size, neq, nz)
    b: torch.Tensor of shape (batch_size, neq)
    G: torch.Tensor of shape (batch_size, nineq, nz)
    h: torch.Tensor of shape (batch_size, nineq)
    rho: float
    
    Returns:
    z: primary solution of shape (batch_size, nz)
    lam: lagrange multiplier of shape (batch_size, neq)
    
    solve
    min_z 1/2 z^T H z + g^T
    s.t. 
    Az = b
    Gz <= h
    
    Move inequality constraint to objective function using penalty method
    min _z, lambda L = 1/2 z^T H z + g^T z + rho/2 (max(0, Gz - h))^2
    s.t. 
    Az = b
    
    Use augmented lagrangian and construct KKT matrix, then solve linear problem. 
    L = 1/2 z^T H z + g^T z + rho/2 (max(0, Gz - h))^2 + lambda^T (Az - b)
    
    KKT condition 
    H z + g + A^T lambda + G^T mu + rho * G^T (Gz-h) = 0 (stationarity)
    Az -b = 0 (primal feasibility)
    
    Above KKT condition reduces problem to linear problem
    Px= q
    P = 
    [H+rho*G^TG, A^T;
     A, 0]
    q = [-g+rho*G^Th; b]
    """
    
    # get shapes
    batch_size = H.shape[0]
    nz = H.shape[1]
    neq = A.shape[1]
    
    # construct KKT matrix
    P = torch.zeros((batch_size, nz+neq, nz+neq), device=H.device)
    P[:, :nz, :nz] = H+rho*torch.bmm(G.transpose(1,2), G)
    P[:, :nz, nz:nz+neq] = A.transpose(1,2)
    P[:, nz:nz+neq, :nz] = A
    
    # construct k
    q = torch.zeros((batch_size, nz+neq), device=H.device)
    q[:, :nz] = -g + torch.bmm(rho*G.transpose(1,2), h.unsqueeze(-1)).squeeze(-1)
    q[:, nz:nz+neq] = b
    
    # solve KKT system Px = q
    # LU factorization based linear solver
    res = torch.linalg.solve(P, q)
    
    z = res[:, :nz]
    lam = res[:, nz:nz+neq]
    
    return z, lam

# def qp_solver_cp(
#     H:cp.ndarray, 
#     g:cp.ndarray, 
#     A:cp.ndarray, 
#     b:cp.ndarray, 
#     G:cp.ndarray, 
#     h:cp.ndarray, 
#     rho:float)->Tuple[cp.ndarray, cp.ndarray]: 
#     """
#     Arguments:
#     H: torch.Tensor of shape (batch_size, nz, nz)
#     g: torch.Tensor of shape (batch_size, nz)
#     A: torch.Tensor of shape (batch_size, neq, nz)
#     b: torch.Tensor of shape (batch_size, neq)
#     G: torch.Tensor of shape (batch_size, nineq, nz)
#     h: torch.Tensor of shape (batch_size, nineq)
#     rho: float
    
#     solve
#     min_z 1/2 z^T H z + g^T
#     s.t. 
#     Az = b
#     Gz <= h
    
#     Move inequality constraint to objective function using penalty method
#     min _z, lambda L = 1/2 z^T H z + g^T z + rho/2 (max(0, Gz - h))^2
#     s.t. 
#     Az = b
    
#     Use augmented lagrangian and construct KKT matrix, then solve linear problem. 
#     L = 1/2 z^T H z + g^T z + rho/2 (max(0, Gz - h))^2 + lambda^T (Az - b)
    
#     KKT condition 
#     H z + g + A^T lambda + G^T mu + rho * G^T (Gz-h) = 0 (stationarity)
#     Az -b = 0 (primal feasibility)
    
#     Above KKT condition reduces problem to linear problem
#     Px= q
#     P = 
#     [H+rho*G^TG, A^T;
#      A, 0]
#     q = [-g+rho*G^Th; b]
#     """
    
#     # get shapes
#     batch_size = H.shape[0]
#     nz = H.shape[1]
#     neq = A.shape[1]
    
#     # construct KKT matrix
#     P = cp.zeros((batch_size, nz+neq, nz+neq))
#     P[:, :nz, :nz] = H + rho * (G.transpose(0,2,1) @ G)
#     P[:, :nz, nz:nz+neq] = A.transpose(0,2,1)
#     P[:, nz:nz+neq, :nz] = A
    
#     # construct k
#     q = cp.zeros((batch_size, nz+neq))
#     q[:, :nz] = -g + (rho * (G.transpose(0,2,1) @ h[:, :, None])).squeeze(-1)
#     q[:, nz:nz+neq] = b
    
#     # solve KKT system Px = q
#     res = cp.linalg.solve(P, q)
    
#     z = res[:, :nz]
#     lam = res[:, nz:nz+neq]
    
#     return z, lam

# # test functions #

# def test_cupy():
#     # set seed
#     cp.random.seed(0)
    
#     # construct qp 
#     batch_size = 4096
#     horizon_length = 10
#     nz = 12*horizon_length
#     nineq = 14*horizon_length
#     neq = 2*horizon_length
#     rho = 10.0
    
#     H = cp.random.randn(batch_size, nz, nz)
#     g = cp.random.randn(batch_size, nz)
#     A = cp.random.randn(batch_size, neq, nz)
#     b = cp.random.randn(batch_size, neq)
#     G = cp.random.randn(batch_size, nineq, nz)
#     h = cp.random.randn(batch_size, nineq)
    
#     z, _ = qp_solver_cp(H, g, A, b, G, h, rho)
#     batch_idx = 0
#     z_prime = z[:, :12]
#     print("mpc solution: ", z_prime[batch_idx])

def test_torch():
    from time import time
    
    # set seed
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # construct qp 
    batch_size = 4096
    horizon_length = 10
    nz = 12*horizon_length
    nineq = 14*horizon_length
    neq = 2*horizon_length
    rho = 10.0

    H = torch.randn(batch_size, nz, nz, device=device)
    g = torch.randn(batch_size, nz, device=device)
    A = torch.randn(batch_size, neq, nz, device=device)
    b = torch.randn(batch_size, neq, device=device)
    G = torch.randn(batch_size, nineq, nz, device=device)
    h = torch.randn(batch_size, nineq, device=device)
    
    t0 = time()
    z, _ = qp_solver_torch(H, g, A, b, G, h, rho)
    print("time took to solve: ", 1000*(time()-t0), " ms")
    
    batch_idx = 0
    z_prime = z[:, :12]
    print("mpc solution: ", z_prime[batch_idx])
    
    cost = (0.5* z.unsqueeze(2).transpose(1,2) @ H @ z.unsqueeze(2)).squeeze(-1) + (g.unsqueeze(2).transpose(1,2) @ z.unsqueeze(2)).squeeze(-1)
    print("cost: ", cost[batch_idx])
    
    A_prime = A[:, :2, :12]
    b_prime = b[:, :2]
    G_prime = G[:, :14, :12]
    h_prime = h[:, :14]
    equality_constraints_violation = (A_prime @ z_prime.unsqueeze(2) - b_prime.unsqueeze(2)).squeeze(2)
    inequality_constraints_violation = torch.max(G_prime @ z_prime.unsqueeze(2) - h_prime.unsqueeze(2), torch.zeros_like(G_prime @ z_prime.unsqueeze(2) - h_prime.unsqueeze(2))).squeeze(2)
    # print("equality constraints_violation: ", equality_constraints_violation[batch_idx])
    # print("inequality constraints_violation: ", inequality_constraints_violation[batch_idx])

if __name__ == "__main__":
    test_torch()
    # test_cupy()