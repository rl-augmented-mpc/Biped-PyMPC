import casadi
import numpy as np

from biped_pympc.casadi.srbd_centroidal_model import SingleRigidBodyDynamics

class SRBD_Constraints:
    """
    Form symbolic QP matrices based on SRBD MPC formulation.
    """
    def __init__(self, horizon:int=10):
        self.horizon = horizon
        self.srbd_model = SingleRigidBodyDynamics(integrator='rk4')
        self.srbd_dynamics = self.srbd_model.discrete_dynamics
        
        self.equality_function = self._equality_constraints()
        self.inequality_function = self._inequality_constraints()
        
        self.qp_mat_former = self._qp_matrices()
        
    def _qp_matrices(self):
        """
        Form symbolic QP matrices and right hand side of form:
        min 0.5 * z.T @ H @ z + f.T @ z
        s.t. A @ z = b
        and G @ z <= d
        where z = [x0, x1, ..., xN, u0, u1, ..., uN-1]
        """
        # system parameters
        num_x = 12
        num_u = 12
        F_max = 500.0
        
        # input time-variant quantities
        dt = casadi.SX.sym('dt', 1)
        m = casadi.SX.sym('m', 1)
        mu = casadi.SX.sym('mu', 1)
        
        # operating point
        R_body = casadi.SX.sym('R_body', 3, 3)
        I_world = casadi.SX.sym('I_world_inv', 3, 3)
        body_pos = casadi.SX.sym('body_pos', 3)
        left_foot_pos = casadi.SX.sym('left_foot_pos', 3)
        right_foot_pos = casadi.SX.sym('right_foot_pos', 3)

        # residual
        residual_lin_accel = casadi.SX.sym('residual_lin_accel', 3)
        residual_ang_accel = casadi.SX.sym('residual_ang_accel', 3)

        # reference
        contact_table = casadi.SX.sym('contact_table', self.horizon, 2)
        x_ref = casadi.SX.sym('x_ref', self.horizon*num_x) # reference trajectory during horizon
        
        # weight matrices
        Q = casadi.SX.sym('Q', num_x)
        R = casadi.SX.sym('R', num_u)
        Q_mat = casadi.diag(Q) # (num_x, num_x)
        R_mat = casadi.diag(R) # (num_u, num_u)
        
        # variables
        x0 = casadi.SX.sym('x0', num_x)
        x = casadi.SX.sym('x', num_x*self.horizon)
        u = casadi.SX.sym('u', num_u*self.horizon)
        
        # quadratic cost
        cost = 0.5*(x - x_ref).T @ casadi.diagcat(*([Q_mat for _ in range(self.horizon)])) @ (x - x_ref) \
            + 0.5*(u.T @ casadi.diagcat(*([R_mat for _ in range(self.horizon)])) @ u)
        H, _ = casadi.hessian(cost, casadi.vertcat(x, u))
        grad = casadi.jacobian(cost, casadi.vertcat(x, u)).T
        f = grad - H @ casadi.vertcat(x, u)
        
        # constraints matrices
        A, b = self.equality_function(x, u, x0, body_pos, left_foot_pos, right_foot_pos, R_body, m, I_world, dt, residual_lin_accel, residual_ang_accel)
        G, d = self.inequality_function(x, u, mu, F_max, contact_table)
        
        qp_former = casadi.Function(
            'qp_former',
            [x0, x, u, x_ref, dt, m, mu, R_body, I_world, body_pos, left_foot_pos, right_foot_pos, contact_table, Q, R, residual_lin_accel, residual_ang_accel],
            [H, f, A, b, G, d]
        )
        
        return qp_former
    
    def _equality_constraints(self):
        """
        Form equality matrices and right hand side. 
        General nequality constraints are in the form:
        l_eq = 0
        
        Linearize at the current state gives
        Az = b form
        where A = jacobian(l_eq, z) and b = -l_eq(z)
        
        equality includes
            - dynamics constraint
            - x moment = 0 (no actuation capability due to lack of ankle roll joint)
        
        Total number of equality constraints is 12 * horizon + 2*horizon = 14*horizon.
        """
        dt = casadi.SX.sym("dt", 1)
        
        x0 = casadi.SX.sym("x0", 12)
        x = casadi.SX.sym("x", 12 * self.horizon) # x1 - xN
        x_cat = casadi.vertcat(x0, x)
        u = casadi.SX.sym("u", 12 * self.horizon) # u0 - uN-1
        z = casadi.vertcat(x, u)
        
        p_body = casadi.SX.sym("p_body", 3)
        p_foot1 = casadi.SX.sym("p_foot1", 3)
        p_foot2 = casadi.SX.sym("p_foot2", 3)
        R_body = casadi.SX.sym("R_body", 3, 3)
        m = casadi.SX.sym("m", 1)
        I_world = casadi.SX.sym("I", 3, 3)
        residual_lin_accel = casadi.SX.sym("residual_lin_accel", 3)
        residual_ang_accel = casadi.SX.sym("residual_ang_accel", 3)
        params = casadi.vertcat(p_body, casadi.reshape(R_body, (9, 1)), p_foot1, p_foot2, m, casadi.reshape(I_world, (9, 1)), residual_lin_accel, residual_ang_accel)
        
        eq_constraints = []
        
        # dynamics constraint
        for i in range(self.horizon):
            x_i = x_cat[i*12:(i+1)*12] # x0, x1, .., xN-1
            u_i = u[i*12:(i+1)*12] # u0, u1, .., uN-1
            x_next = x_cat[(i+1)*12:(i+2)*12] # x1, x2, .., xN
            x_predicted = self.srbd_dynamics(x_i, u_i, params, dt)
            dynamics_constraint = x_next - x_predicted # x_i+1 - (A*x_i + B*u_i) = 0
            eq_constraints.append(dynamics_constraint)
        
        # x moment = 0
        for i in range(self.horizon):
            u_i = u[i*12:(i+1)*12]
            mx_left = u_i[6]
            mx_right = u_i[9]
            
            eq_constraints.append(mx_left) # Mx_left = 0
            eq_constraints.append(mx_right) # Mx_right = 0
            
        
        eq_constraints = casadi.vertcat(*eq_constraints)
        A = casadi.jacobian(eq_constraints, z)
        b = A @ z - eq_constraints
        
        return casadi.Function("equality_constraints", [x, u, x0, p_body, p_foot1, p_foot2, R_body, m, I_world, dt, residual_lin_accel, residual_ang_accel], [A, b],)
    
    def _inequality_constraints(self):
        """
        Form inequality matrices and right hand side. 
        General inequality constraints are in the form:
        l_ineq <= 0
        
        Linearize at the current state gives
        Gz <= d form
        where G = jacobian(l_ineq, z)
        and d = -l_ineq(z)
        
        inequality includes
         - friction pyramid
         - line contact constraint
         
        Total number of inequality constraints is 12 * horizon. 
        """
        lt = 0.07
        lh = 0.04
        
        x = casadi.SX.sym("x", 12 * self.horizon)
        u = casadi.SX.sym("u", 12 * self.horizon)
        z = casadi.vertcat(x, u)
        
        mu = casadi.SX.sym("mu", 1)
        f_max = casadi.SX.sym("f_max", 1)
        contact_table = casadi.SX.sym("contact_table", self.horizon, 2)
        
        ineq_constraints = []
        
        ## 1. friction pyramid
        # -mu*fz - fx <= 0
        # mu*fz - fx <= 0
        # -mu*fz - fy <= 0
        # fy - mu*fz <= 0
        ## 2. force saturation
        # -fz <= 0
        # fz - f_max*contact <= 0
        ## 3. line contact constraint
        # [0, 0, -lt]*F -  [0, 1, 0]*M <= 0 
        # -[0, 0, lh]*F + [0, 1, 0]*M <= 0

        for i in range(self.horizon):
            u_i = u[i*12:(i+1)*12]
            f1 = u_i[0:3]
            f2 = u_i[3:6]
            m1 = u_i[6:9]
            m2 = u_i[9:12]
            
            # friction pyramid left foot
            ineq1 = -f1[0] - mu*f1[2]
            ineq2 = f1[0] - mu*f1[2]
            ineq3 = -f1[1] - mu*f1[2]
            ineq4 = f1[1] - mu*f1[2]
            
            # line contact constraint left foot
            ineq5 = -lt*f1[2] - m1[1]
            ineq6 = -lh*f1[2] + m1[1]
            
            # force saturation
            ineq7 = -f1[2]
            ineq8 = f1[2] - f_max*contact_table[i, 0]
            
            # friction pyramid right foot
            ineq9 = -f2[0] - mu*f2[2]
            ineq10 = f2[0] - mu*f2[2]
            ineq11 = -f2[1] - mu*f2[2]
            ineq12 = f2[1] - mu*f2[2]
            
            # line contact constraint right foot
            ineq13 = -lt*f2[2] - m2[1]
            ineq14 = -lh*f2[2] + m2[1]
            
            # force saturation
            ineq15 = -f2[2]
            ineq16 = f2[2] - f_max*contact_table[i, 1]
            
            ineq_constraints.extend([ineq1, ineq2, ineq3, ineq4, ineq5, ineq6, ineq7, ineq8,
                                    ineq9, ineq10, ineq11, ineq12, ineq13, ineq14, ineq15, ineq16])
        
        ineq_constraints = casadi.vertcat(*ineq_constraints)
        G = casadi.jacobian(ineq_constraints, z)
        d = G @ z - ineq_constraints
        return casadi.Function("inequality_constraints", [x, u, mu, f_max, contact_table], [G, d])



if __name__ == "__main__":
    import os
    from os import system
    import time

    num_horizon = 10
    srbd_constraint = SRBD_Constraints(num_horizon)
    
    print("num_instructions: ", srbd_constraint.qp_mat_former.n_instructions())
    
    # Test the function
    
    # dynamics parameters
    dt = 0.001*40 # mpc sampling time
    m = 13.5
    friction_coef = 0.5
    R_b = np.eye(3)
    I_w = np.array([[0.5413, 0.0, 0.0],
                        [0.0, 0.5200, 0.0],
                        [0.0, 0.0, 0.0691]])
    body_pos = np.array([0.0, 0.0, 0.5])
    left_foot_pos = np.array([0.1, 0.05, 0.0])
    right_foot_pos = np.array([0.1, -0.05, 0.0])
    contact_table = np.ones((num_horizon,2))
    
    # MPC tracking weight
    Q = np.array([50, 50, 10,  10, 10, 100,  10, 10, 10,  10, 10, 10])
    R = np.array([1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1,   1e-1, 1e-1, 1e-1, 1e-1, 1e-1, 1e-1])

    # linear and angular residual accelerations (to simulate model mismatch)
    residual_lin_accel = np.array([0.0, 0.0, 0.0])
    residual_ang_accel = np.array([0.0, 0.0, 0.0])
    
    # form state
    z = np.zeros(((12+12)*num_horizon))
    for i in range(num_horizon):
        z[i*12+5] = 0.5
        z[12*num_horizon +i*12+2] = m*9.81/2
        z[12*num_horizon+i*12+5] = m*9.81/2
        z[12*num_horizon +i*12+6:12*num_horizon +i*12+9] = np.cross(left_foot_pos-body_pos, z[12*num_horizon +i*12:12*num_horizon +i*12+3])
        z[12*num_horizon +i*12+9:12*num_horizon +i*12+12] = np.cross(right_foot_pos-body_pos, z[12*num_horizon +i*12+3:12*num_horizon +i*12+6])
    x = z[:12*num_horizon]
    u = z[12*num_horizon:]
    
    # reference trajectory
    x_ref = np.zeros((12*num_horizon))
    for i in range(num_horizon):
        x_ref[i*12+5] = 0.55
    
    # initial state
    x0 = np.zeros((12))
    x0[5] = 0.55
    
    H, f, A, b, G, d = srbd_constraint.qp_mat_former(
        x0, x, u, x_ref, dt, m, friction_coef, R_b, I_w, 
        body_pos, left_foot_pos, right_foot_pos, contact_table, 
        Q, R, residual_lin_accel, residual_ang_accel
        )
    
    print("H: ", H.shape)
    print("f: ", f.shape)
    print("A: ", A.shape)
    print("b: ", b.shape)
    print("G: ", G.shape)
    print("d: ", d.shape)
    
    # non zero numbers
    print("H density", H.nnz()/(H.shape[0] * H.shape[1]))
    print("f density", f.nnz()/(f.shape[0] * f.shape[1]))
    print("A density", A.nnz()/(A.shape[0] * A.shape[1]))
    print("b density", b.nnz()/(b.shape[0] * b.shape[1]))
    print("G density", G.nnz()/(G.shape[0] * G.shape[1]))
    print("d density", d.nnz()/(d.shape[0] * d.shape[1]))
    
    # get CCS triplets 
    Hzz = H.nonzeros()
    H_col_point, H_row_ind = H.sparsity().get_ccs()
    
    Azz = A.nonzeros()
    A_col_point, A_row_ind = A.sparsity().get_ccs()
    
    Gzz = G.nonzeros()
    G_col_point, G_row_ind = G.sparsity().get_ccs()
     
    
    # save the function
    function_path = os.path.join("biped_pympc", "casadi", "function", "srbd_qp_mat.casadi")
    cusadi_function_path = os.path.join("biped_pympc", "cusadi", "src", "casadi_functions", "srbd_qp_mat.casadi")
    srbd_constraint.qp_mat_former.save(function_path)
    system(f"mv {function_path} {os.path.join('biped_pympc/cusadi/src/casadi_functions', 'srbd_qp_mat.casadi')}")
    print(f"saved function to {cusadi_function_path}")

    # # code generation 
    # name = "qp_former"
    # cname = srbd_constraint.qp_mat_former.generate()
    # oname_O1 = name + '_O1.so'
    # print('Compiling with O1 optimization: ', oname_O1)
    # t1 = time.time()
    # system('gcc -fPIC -shared -O1 ' + cname + ' -o ' + oname_O1)
    # t2 = time.time()
    # print('compile time = ', (t2-t1)*1e3, ' ms')

    # # move to codegen directory
    # codegen_path = os.path.join("biped_pympc", "casadi", "codegen")
    # if not os.path.exists(codegen_path):
    #     os.makedirs(codegen_path)
    # system(f"mv {cname} {codegen_path}")
    # system(f"mv {oname_O1} {codegen_path}")

    # codegen_function = casadi.external(name, os.path.join(codegen_path, oname_O1))
    # print(codegen_function)