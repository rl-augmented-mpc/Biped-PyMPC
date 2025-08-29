from typing import Literal
import casadi
import numpy as np

class SingleRigidBodyDynamics:
    def __init__(self, integrator:Literal["euler", "rk4"]="euler")->None:
        
        com_position_x = casadi.SX.sym("com_position_x", 1, 1)
        com_position_y = casadi.SX.sym("com_position_y", 1, 1)
        com_position_z = casadi.SX.sym("com_position_z", 1, 1)
        
        roll = casadi.SX.sym("roll", 1, 1)
        pitch = casadi.SX.sym("pitch", 1, 1)
        yaw = casadi.SX.sym("yaw", 1, 1)
        
        com_velocity_x = casadi.SX.sym("com_velocity_x", 1, 1)
        com_velocity_y = casadi.SX.sym("com_velocity_y", 1, 1)
        com_velocity_z = casadi.SX.sym("com_velocity_z", 1, 1)
        
        omega_x = casadi.SX.sym("omega_x", 1, 1)
        omega_y = casadi.SX.sym("omega_y", 1, 1)
        omega_z = casadi.SX.sym("omega_z", 1, 1)
        
        self.states = casadi.vertcat(
            roll, 
            pitch,
            yaw,
            com_position_x,
            com_position_y,
            com_position_z,
            omega_x,
            omega_y,
            omega_z,
            com_velocity_x,
            com_velocity_y,
            com_velocity_z,
        )
        
        foot_force_left = casadi.SX.sym("foot_force_left", 3, 1)
        foot_force_right = casadi.SX.sym("foot_force_right", 3, 1)
        foot_moment_left = casadi.SX.sym("foot_moment_left", 3, 1)
        foot_moment_right = casadi.SX.sym("foot_moment_right", 3, 1)
        
        self.inputs = casadi.vertcat(
            foot_force_left,
            foot_force_right,
            foot_moment_left,
            foot_moment_right
        )
        
        # linearization points
        com_position = casadi.SX.sym("com_position", 3, 1)
        com_rotation = casadi.SX.sym("com_rotation", 9, 1)
        foot_position_left = casadi.SX.sym("foot_position_left", 3, 1)
        foot_position_right = casadi.SX.sym("foot_position_right", 3, 1)
        mass = casadi.SX.sym("mass", 1, 1)
        inertia = casadi.SX.sym("inertia", 9, 1)
        lin_accel_residual = casadi.SX.sym("lin_accel_residual", 3, 1)
        ang_accel_residual = casadi.SX.sym("ang_accel_residual", 3, 1)
        self.params = casadi.vertcat(
            com_position,
            com_rotation,
            foot_position_left,
            foot_position_right,
            mass,
            inertia, 
            lin_accel_residual,
            ang_accel_residual
        )
        self.param_dim = self.params.size()[0]
        
        state_dot = self.forward_dynamics(self.states, self.inputs, self.params)
        self.func_dynamics = casadi.Function(
            "centroidal_dynamics",
            [self.states, self.inputs, self.params],
            [state_dot]
        )
        
        if integrator == "rk4":
            self.discrete_dynamics = self.rk4_integrator()
        elif integrator == "euler":
            self.discrete_dynamics = self.forward_euler_integrator()
        
    def forward_euler_integrator(self):
        """
        Forward Euler integrator.
        """
        state = casadi.SX.sym("state", 12, 1)
        inputs = casadi.SX.sym("inputs", 12, 1)
        params = casadi.SX.sym("params", self.param_dim, 1)
        dt = casadi.SX.sym("dt", 1, 1)
        
        state_next = state + dt * self.func_dynamics(state, inputs, params)
        
        return casadi.Function(
            "forward_euler_integrator",
            [state, inputs, params, dt],
            [state_next]
        )
    
    def rk4_integrator(self):
        """
        Runge-Kutta 4th order integrator.
        """
        state = casadi.SX.sym("state", 12, 1)
        inputs = casadi.SX.sym("inputs", 12, 1)
        params = casadi.SX.sym("params", self.param_dim, 1)
        dt = casadi.SX.sym("dt", 1, 1)
        
        k1 = self.func_dynamics(state, inputs, params)
        k2 = self.func_dynamics(state + dt/2 * k1, inputs, params)
        k3 = self.func_dynamics(state + dt/2 * k2, inputs, params)
        k4 = self.func_dynamics(state + dt * k3, inputs, params)
        
        state_next = state + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        
        return casadi.Function(
            "rk4_integrator",
            [state, inputs, params, dt],
            [state_next]
        )
        
    def forward_dynamics(self, states, inputs, params):
        """
        Forward dynamics.
        """
        # Extracting parameters
        body_position = params[0:3]
        body_rotation = casadi.reshape(params[3:12], (3, 3))
        foot_position_left = params[12:15]
        foot_position_right = params[15:18]
        mass = params[18]
        inertia = casadi.reshape(params[19:28], (3, 3))
        residual_lin_accel = params[28:31]
        residual_ang_accel = params[31:34]
        gravity = casadi.SX([0, 0, -9.81])
        
        # Extracting states
        euler = states[0:3]
        com_position = states[3:6]
        com_omega = states[6:9]
        com_velocity = states[9:12]
        
        # Extracting inputs
        foot_force_left = inputs[0:3]
        foot_force_right = inputs[3:6]
        foot_moment_left = inputs[6:9]
        foot_moment_right = inputs[9:12]
        
        
        # dynamics
        inertia_inv = casadi.inv(inertia)
        euler_dot = body_rotation @ com_omega
        com_ang_acc = inertia_inv @ (
                    casadi.skew(foot_position_left-body_position) @ foot_force_left + \
                    casadi.skew(foot_position_right-body_position) @ foot_force_right + \
                    foot_moment_left + foot_moment_right
                    ) + residual_ang_accel
        com_acc = (foot_force_left + foot_force_right) / mass + gravity + residual_lin_accel
        
        return casadi.vertcat(
            euler_dot, 
            com_velocity, 
            com_ang_acc, 
            com_acc
        )


if __name__ == "__main__":
    from biped_pympc.casadi.utils.animation import animate_srbd
    np.set_printoptions(
    linewidth=200,  # set wide enough so rows don't wrap
    suppress=True,  # avoid scientific notation
    precision=3     # limit decimal digits
    )
    
    num_states = 12
    num_inputs = 12
    mu = 0.5
    f_max = 250.0
    dt = 0.001*40
    g = 9.81
    m = 13.5 # mass of the robot
    
    # initial state
    x0 = np.zeros(12)
    x0[5] = 0.55
    
    # params
    mass = np.array([13.5])
    inertia = np.diag([0.5431, 0.52, 0.0691])
    body_pos = np.array([0.0, 0.0, 0.55])
    R_body = np.eye(3)
    left_foot_pos = np.array([0.0, 0.05, 0.0])
    right_foot_pos = np.array([0.0, -0.05, 0.0])
    params = np.concatenate((
        body_pos,
        R_body.flatten(),
        left_foot_pos,
        right_foot_pos,
        mass,
        inertia.flatten()
    ))
    
    # states
    x_current = np.zeros(12) 
    x_current[5] = 0.55 # z
    
    u = np.array([0, 0, m*g/2, 0, 0, m*g/2, 0, 0, 0, 0, 0, 0]) # f1x, f1y, f1z, f2x, f2y, f2z, m1x, m1y, m1z, m2x, m2y, m2z
    u[6:9] = -np.cross(left_foot_pos-body_pos, u[0:3]) # m1
    u[9:12] = -np.cross(right_foot_pos-body_pos, u[3:6]) # m2
    
    
    centroidal_dynamics = SingleRigidBodyDynamics()
    centroidal_discrete_dynamics = centroidal_dynamics.discrete_dynamics
    
    
    # steps = 100 #dt=0.02*100 = 2s
    # t_bin = np.linspace(0, steps*dt, steps)
    # poses = []
    # forces_left = []
    # forces_right = []
    # moments_left = []
    # moments_right = []
    # poses.append(x_current[:6])
    # forces_left.append(u[0:3])
    # forces_right.append(u[3:6])
    # moments_left.append(u[6:9])
    # moments_right.append(u[9:12])
    # for t in t_bin:
    #     x_next = centroidal_discrete_dynamics(x_current, u, params, dt).full().flatten()
    #     # print("state at t = {:.3f}: ".format(t), x_next)
    #     x_current = x_next
    #     poses.append(x_current[:6])
    #     forces_left.append(u[0:3])
    #     forces_right.append(u[3:6])
    #     moments_left.append(u[6:9])
    #     moments_right.append(u[9:12])
        
    # animate_srbd(
    #     poses=poses, 
    #     forces_L=forces_left,
    #     forces_R=forces_right,
    #     moments_L=forces_left,
    #     moments_R=forces_right,
    #     foot_L=(0.0, 0.05, 0.0),
    #     foot_R=(0.0, -0.05, 0.0),
    #     l=0.2, w=0.3, h=1.0, 
    #     force_scale=0.005, 
    #     moment_scale=0.005,
    # )